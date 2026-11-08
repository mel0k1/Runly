use runly::prelude::*;
use std::fs;
use std::path::PathBuf;

fn temp_csv(name: &str, content: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("runly_{}_{}", std::process::id(), name));
    fs::write(&path, content).expect("write temp csv");
    path
}

#[test]
fn csv_loader_reads_header_and_last_column_label() -> Result<()> {
    let path = temp_csv(
        "basic.csv",
        "sepal,petal,species\n5.1,1.4,0\n4.9,1.4,0\n6.7,5.2,1\n6.3,6.0,2\n",
    );
    let ds = load_csv(&path)?;
    assert_eq!(ds.dim, 2);
    assert_eq!(ds.len(), 4);
    assert_eq!(ds.num_classes, 3);
    assert_eq!(ds.y, vec![0, 0, 1, 2]);
    assert!((ds.x[0] - 5.1).abs() < 1e-6);
    assert!((ds.x[5] - 5.2).abs() < 1e-6);
    fs::remove_file(&path)?;
    Ok(())
}

#[test]
fn csv_loader_handles_quoted_fields_and_named_label() -> Result<()> {
    let path = temp_csv(
        "quoted.csv",
        "name,value,tag\n\"5.5\",1.5,0\n\"6.5\",2.5,1\n\"7.5\",3.0,0\n",
    );
    let ds = load_csv_named(&path, "tag")?;
    assert_eq!(ds.dim, 2);
    assert_eq!(ds.len(), 3);
    assert_eq!(ds.y, vec![0, 1, 0]);
    assert!((ds.x[0] - 5.5).abs() < 1e-6);
    assert!((ds.x[4] - 7.5).abs() < 1e-6);
    assert!((ds.x[5] - 3.0).abs() < 1e-6);
    fs::remove_file(&path)?;
    Ok(())
}

#[test]
fn csv_loader_config_selects_delimiter_and_label_column() -> Result<()> {
    let path = temp_csv("semi.csv", "0.5;0.25;10\n1.5;0.75;11\n");
    let ds = load_csv_with(
        &path,
        CsvConfig {
            delimiter: ';',
            header: false,
            label: Some(2),
        },
    )?;
    assert_eq!(ds.dim, 2);
    assert_eq!(ds.len(), 2);
    assert_eq!(ds.y, vec![10, 11]);
    assert_eq!(ds.num_classes, 12);
    fs::remove_file(&path)?;
    Ok(())
}

#[test]
fn csv_loader_rejects_ragged_rows_and_bad_numbers() {
    let path = temp_csv("ragged.csv", "a,b,l\n1.0,2.0,0\n3.0,4.0,1,9\n");
    let err = load_csv(&path).unwrap_err();
    assert!(err.to_string().contains("row 2"), "{err}");
    fs::remove_file(&path).unwrap();

    let path = temp_csv("nonnum.csv", "a,b,l\n1.0,x,0\n3.0,4.0,1\n");
    let err = load_csv(&path).unwrap_err();
    assert!(err.to_string().contains("cannot parse"), "{err}");
    fs::remove_file(&path).unwrap();
}
