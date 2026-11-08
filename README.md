# Runly

[![CI](https://github.com/mel0k1/Runly/actions/workflows/ci.yml/badge.svg)](https://github.com/mel0k1/Runly/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Нейросети на Rust. Runly собирает многослойный перцептрон из линейных слоёв и активаций, считает градиенты через автоград и сохраняет обученные веса в safetensors. Тензорные вычисления выполняет candle от Hugging Face, сверху у него небольшой API с минимальным количеством лишних слов.

Английская версия README: [README_EN.md](README_EN.md)

## Зачем ещё одна библиотека

Обёрток над Python много, а на Rust нейросеть с нуля собрать не так легко: API candle низкоуровневый, а примеры в документации рассчитаны на тех, кто уже знает, чего хочет. Runly делает минимальный набор для обучения MLP, который читается за один вечер.

Это неплохой первый проект, если хочется понять, как автоград устроен внутри. Код занимает несколько сотен строк, и каждый шаг обучения виден в трёх строчках примера.

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
    let device = Device::Cpu;
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

## Что внутри

| Модуль | Что делает |
|--------|------------|
| `model` | последовательность Linear и активаций: relu, tanh, sigmoid, gelu, softmax |
| `optim` | Adam и SGD с моментумом, плюс AdamW из candle |
| `loss` | кросс-энтропия, MSE, Huber, бинарная кросс-энтропия с логитами |
| `data` | генераторы xor и two spirals, train/test split, мини-батчи |
| `gradcheck` | сверка аналитических градиентов с численными |
| `metrics` | argmax и accuracy |
| `mnist` | загрузка MNIST по фиче `mnist` |

## Задачи, на которых проверялась библиотека

| Задача | Сеть | Результат |
|--------|------|-----------|
| XOR | 2-16-2, tanh | accuracy 1.000, loss 0.002 за 300 эпох |
| two spirals | 2-64-64-2, relu | test accuracy 1.000 за 800 эпох |
| MNIST | 784-128-10, relu | test accuracy 0.972 за 5 эпох |

MNIST учится примерно за минуту на двух ядрах процессора.

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

Признаки f32, метки u32. Дальше обычный цикл обучения из примера выше.

## Сохранение и загрузка

```rust
model.save("model.safetensors")?;
let restored = Model::load("model.safetensors", &device)?;
```

В файл попадают веса и описание архитектуры, поэтому load возвращает готовую модель, которую можно дообучать.

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

Восемь тестов: шаги Adam и SGD сверяются с ручным расчётом, градиент-чек проходит на двух архитектурах, xor и спирали реально обучаются, веса сохраняются и восстанавливаются без искажений. GitHub Actions на каждый push гоняет fmt, clippy и тесты.

## MNIST

```bash
cargo run --release --features mnist --example mnist
```

Фича `mnist` добавляет загрузчик: при первом запуске он скачает четыре файла примерно на 11 МБ в папку data, распакует и соберёт тензоры.

## Планы

Свёрточные слои, загрузка CSV, GPU через cuda-фичу candle, сохранение в компактный формат. Если чего-то из этого не хватает, issues открыты.

## Лицензия

Apache-2.0, текст в файле LICENSE.
