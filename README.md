# Runly

[![CI](https://github.com/mel0k1/Runly/actions/workflows/ci.yml/badge.svg)](https://github.com/mel0k1/Runly/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Нейросети на Rust. Runly собирает сети из линейных и свёрточных слоёв, активаций и пулинга, считает градиенты через автоград и сохраняет обученные веса в safetensors. Тензорные вычисления выполняет candle от Hugging Face, сверху у него небольшой API с минимальным количеством лишних слов.

Английская версия README: [README_EN.md](README_EN.md)

## Быстрый старт

Нужен только Rust:

```bash
git clone https://github.com/mel0k1/Runly.git
cd Runly
cargo run --release --example spirals
```

Пример обучает сеть 2-64-64-2 на задаче two spirals. Вывод такой:

```text
epoch   1  loss 1.0616  test accuracy 0.650
epoch 100  loss 0.2502  test accuracy 0.950
epoch 300  loss 0.0147  test accuracy 0.988
epoch 800  loss 0.0033  test accuracy 1.000
train accuracy 1.000
test accuracy 1.000
```

## Пример кода

```rust
use runly::prelude::*;

fn main() -> candle::Result<()> {
    let device = device()?;
    let data = xor(64, 0.15, 42);
    let (x, y) = data.tensors(&device)?;

    let mut model = Model::new(&device);
    model.linear(2, 16)?;
    model.tanh();
    model.linear(16, 2)?;

    let mut opt = Adam::new_lr(model.vars(), 0.02)?;
    for _ in 0..300 {
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
    }
    println!("accuracy {:.3}", accuracy(&model.forward(&x)?, &y)?);
    Ok(())
}
```

forward строит граф вычислений, cross_entropy возвращает скаляр, backward_step считает градиенты и обновляет параметры. Сессий, этапов компиляции и отдельных функций train в библиотеке нет.

## Планировщики learning rate

Планировщик выдаёт learning rate для текущего шага и двигает внутренний счётчик, apply заодно выставляет его оптимизатору:

```rust
let mut opt = Adam::new_lr(model.vars(), 0.02)?;
let mut sched = CosineAnnealingLR::new(0.02, 300, 1e-4)?;
for epoch in 0..300 {
    sched.apply(&mut opt);
    let logits = model.forward(&x)?;
    let loss = cross_entropy(&logits, &y)?;
    opt.backward_step(&loss)?;
}
```

Внутри StepLR, ExponentialLR и CosineAnnealingLR: первый режет lr в gamma раз каждые step_size шагов, второй умножает на gamma на каждом шаге, третий идёт по косинусу от базового значения до eta_min за t_max шагов. Текущее значение возвращает метод lr(), планировщик работает с любым оптимизатором, реализующим трейт Optimizer.

Warmup оборачивает любой планировщик: первые warmup_steps шагов lr растёт линейно до peak, потом управление переходит к внутреннему расписанию:

```rust
let cosine = CosineAnnealingLR::new(0.02, 280, 1e-4)?;
let mut sched = Warmup::new(cosine, 20, 0.02)?;
```

OneCycleLR поднимает lr от initial_lr до peak за warmup-долю от total_steps и по косинусу спускает до final_lr, весь цикл рассчитан на один прогон обучения:

```rust
let mut sched = OneCycleLR::new(0.02, 300, 0.3, 0.001, 1e-4)?;
```

examples/sched.rs обучает xor с warmup плюс косинусом и с one cycle, печатая lr по ходу.

## Свёрточные сети

conv2d добавляет свёрточный слой, max_pool и avg_pool уменьшают карты признаков, flatten разворачивает их в вектор перед линейными слоями:

```rust
let mut model = Model::new(&device);
model.conv2d(1, 8, 5, 1, 2)?;
model.relu();
model.max_pool(2);
model.conv2d(8, 16, 5, 1, 2)?;
model.relu();
model.max_pool(2);
model.flatten();
model.linear(16 * 7 * 7, 10)?;
```

Аргументы conv2d это входные каналы, выходные каналы, размер ядра, stride и padding. Вход задаётся формой (batch, каналы, высота, ширина), размер при пулинге должен делиться на ядро нацело. max_pool берёт в окне максимум, avg_pool среднее, оба работают и как отдельные функции над тензором, если нужен пулинг без модели. global_avg_pool усредняет каждую карту целиком, приводит любой вход к форме (N, C, 1, 1) и не требует делить размер нацело, после него перед линейным слоем нужен flatten.

В save/load вместе с весами попадает описание архитектуры вместе со stride и padding, поэтому свёрточная модель восстанавливается из файла целиком и готова к дообучению.

## BatchNorm и dropout

batch_norm ставится после линейного или свёрточного слоя и нормализует активации по батчу, вход может быть (N, C) или (N, C, H, W). dropout выключает случайную долю нейронов при обучении:

```rust
model.conv2d(1, 8, 5, 1, 2)?;
model.batch_norm(8)?;
model.relu();
model.max_pool(2);
model.dropout(0.25);
```

Слой batch_norm держит бегущие среднее и дисперсию: forward считает статистики по батчу и обновляет бегущие. Для инференса и оценки на тесте есть forward_eval — он работает на накопленных статистиках и не обновляет их, а dropout просто выключается:

```rust
let acc = accuracy(&model.forward_eval(&test_x)?, &test_y)?;
```

## CSV

Загрузчик читает числовые таблицы: все колонки, кроме метки, становятся признаками f32, метка это u32. По умолчанию разделитель запятая, первая строка это заголовок, метка лежит в последней колонке:

```rust
let ds = load_csv("iris.csv")?;
let ds = load_csv_named("iris.csv", "species")?;
```

CsvConfig позволяет выбрать разделитель, отключить заголовок и задать колонку с меткой по индексу. Кавычки и запятые внутри кавычек разбираются по правилам CSV, ошибка парсинга называет строку и колонку. Метки должны быть целыми числами от нуля. Пример examples/csv.rs записывает спирали в CSV, читает их обратно и обучает сеть.

## GPU

device() возвращает CUDA-устройство, если библиотека собрана с фичей cuda, и CPU в остальных случаях, поэтому один и тот же код работает в обоих режимах:

```bash
cargo run --release --features cuda --example cnn
```

Нужен установленный CUDA Toolkit: candle компилирует ядро при сборке, поэтому первая компиляция занимает заметно дольше обычной.

## Что внутри

| Модуль | Что делает |
|--------|------------|
| `model` | Linear, Conv2d, batch_norm, max_pool, avg_pool, global_avg_pool, dropout, flatten и активации: relu, tanh, sigmoid, gelu, softmax |
| `optim` | Adam и SGD с моментумом и weight decay, плюс AdamW из candle |
| `sched` | планировщики learning rate: StepLR, ExponentialLR, CosineAnnealingLR, Warmup, OneCycleLR |
| `checkpoint` | трекинг лучшей модели по метрике |
| `loss` | кросс-энтропия, MSE, Huber, бинарная кросс-энтропия с логитами |
| `data` | генераторы xor и two spirals, CSV-загрузчик, train/test split, мини-батчи |
| `gradcheck` | сверка аналитических градиентов с численными |
| `metrics` | argmax и accuracy |
| `mnist` | загрузка MNIST по фиче `mnist` |

## Задачи, на которых проверялась библиотека

| Задача | Сеть | Результат |
|--------|------|-----------|
| XOR | 2-16-2, tanh | accuracy 1.000, loss 0.002 за 300 эпох |
| two spirals | 2-64-64-2, relu | test accuracy 1.000 за 800 эпох |
| MNIST | 784-128-10, relu | test accuracy 0.972 за 5 эпох |
| MNIST CNN | 8-5x5 и 16-5x5 с maxpool | test accuracy 0.985 за 5 эпох |

MLP на MNIST учится примерно за минуту на двух ядрах процессора, свёрточная сеть на тех же ядрах за семь.

## Свои данные

Датасет это плоский вектор признаков и вектор меток:

```rust
let ds = Dataset {
    x: features,
    y: labels,
    dim: 4,
    num_classes: 3,
};
let (x, y) = ds.tensors(&device)?;
```

Признаки f32, метки u32. Дальше обычный цикл обучения из примера выше. Если данные лежат в таблице, проще взять CSV-загрузчик из раздела выше.

## Сохранение и загрузка

```rust
model.save("model.safetensors")?;
let restored = Model::load("model.safetensors", &device)?;
```

В файл попадают веса и описание архитектуры, включая параметры свёрток, поэтому load возвращает готовую модель, которую можно дообучать. save_half пишет веса в f16, файл получается примерно вдвое меньше, а load сам поднимает точность до f32:

```rust
model.save_half("model.safetensors")?;
let restored = Model::load("model.safetensors", &device)?;
```

Во время обучения удобнее хранить не последнюю, а лучшую модель. Checkpointer сравнивает метрику с предыдущей и перезаписывает файл, только когда метрика улучшилась, лучший результат виден через best:

```rust
let mut best = Checkpointer::new("best.safetensors");
for epoch in 0..300 {
    sched.apply(&mut opt);
    let logits = model.forward(&x)?;
    let loss = cross_entropy(&logits, &y)?;
    opt.backward_step(&loss)?;
    best.update(&model, loss.to_dtype(DType::F64)?.to_scalar::<f64>()?)?;
}
let model = Model::load("best.safetensors", &device)?;
```

## Проверка градиентов

Автоград легко написать неправильно, поэтому в тестах он сверяется с численным дифференцированием: градиент каждого параметра считается двумя способами, расхождение не должно выходить за допуски. Свою модель можно проверить той же функцией:

```rust
let reports = check_model(
    &model,
    &x,
    |logits| cross_entropy(logits, &y),
    GradCheckConfig::default(),
)?;
```

## Скорость

examples/bench.rs меряет matmul в f32. На двух ядрах Xeon получилось 140-160 GFLOPS, большую часть скорости даёт крейт gemm с ручной векторизацией внутри candle.

## Тесты и CI

```bash
cargo test
```

Пятьдесят тестов: шаги Adam и SGD с weight decay сверяются с ручным расчётом, планировщики выдают точные расписания и реально двигают lr, градиент-чек проходит на линейных и свёрточных архитектурах, batch_norm и всех видах пулинга, avg_pool и global_avg_pool сверяются с ручным расчётом значений, xor, спирали и свёрточная сеть с batch_norm реально обучаются, веса сохраняются в f32 и f16 и восстанавливаются без искажений, batch_norm корректно переключается между обучением и инференсом, dropout зануляет ожидаемую долю нейронов, CSV-загрузчик разбирает кавычки и сообщает о битых строках, Checkpointer держит лучшую модель. GitHub Actions на каждый push гоняет fmt, clippy и тесты.

## MNIST

```bash
cargo run --release --features mnist --example mnist
cargo run --release --features mnist --example cnn
```

Фича `mnist` добавляет загрузчик: при первом запуске он скачает четыре файла примерно на 11 МБ в папку data, распакует и соберёт тензоры. mnist обучает перцептрон, cnn ту же задачу решает свёрточной сетью.

## Лицензия

Apache-2.0, текст в файле LICENSE.
