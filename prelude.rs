use base64::prelude::*;
use image::{ImageBuffer, Rgb};
use plotters::coord::Shift;
use plotters::prelude::*;
use polars::prelude::*;
use std::io::Cursor;

pub struct PlotWrapper(String);

impl PlotWrapper {
    pub fn evcxr_display(&self) {
        println!("{}", self.0);
    }
}

pub fn my_evcxr_figure<
    Draw: FnOnce(DrawingArea<BitMapBackend, Shift>) -> Result<(), Box<dyn std::error::Error>>,
>(
    size: (u32, u32),
    draw: Draw,
) -> PlotWrapper {
    let mut rgb_data = vec![0u8; (size.0 * size.1 * 3) as usize];
    let root = BitMapBackend::with_buffer(&mut rgb_data, size).into_drawing_area();
    draw(root).expect("Drawing failure");

    // buffer now contains the raw RGB content.
    let mut buf = vec![0u8; (size.0 * size.1 * 3) as usize];

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(size.0, size.1, rgb_data).expect("ERROR!");
    // let img = load_from_memory(&rgb_data).unwrap();
    img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
        .unwrap();

    let b64 = BASE64_STANDARD.encode(buf);

    PlotWrapper(format!(
        "EVCXR_BEGIN_CONTENT image/png\n{b64}\nEVCXR_END_CONTENT"
    ))
}

use std::fmt::Write;

/// Escapes LaTeX special characters (currently only underscores).
fn escape_latex(s: &str) -> String {
    s.replace("_", "\\_")
}

/// Formats an `AnyValue` to display only the underlying data:
/// - Floats are formatted to four decimal places.
/// - Null values become "null".
/// - Lists of f64 are rendered like "[0.3512, 0.2422, 0.3788]".
fn format_any_value(value: &AnyValue) -> String {
    match value {
        AnyValue::Null => "null".to_owned(),
        AnyValue::String(s) => s.to_string(),
        AnyValue::StringOwned(s) => s.to_string(),
        AnyValue::Boolean(b) => b.to_string(),
        AnyValue::Int32(i) => i.to_string(),
        AnyValue::Int64(i) => i.to_string(),
        AnyValue::Float32(f) => {
            format!("{:.4}", f)
        }
        AnyValue::Float64(f) => {
            format!("{:.4}", f)
        }
        AnyValue::List(series) => {
            // Try to treat the list as a list of f64.
            if let Ok(ca) = series.f64() {
                let formatted: Vec<String> = ca
                    .into_iter()
                    .map(|opt| match opt {
                        Some(val) => {
                            format!("{:.4}", val)
                        }
                        None => "null".to_owned(),
                    })
                    .collect();
                format!("[{}]", formatted.join(", "))
            } else {
                // Fallback: use the debug formatting.
                format!("{:?}", series)
            }
        }
        _ => format!("{:?}", value),
    }
}

/// Converts a Polars DataFrame into a LaTeX table using the tabularx environment.
/// The table will be wrapped to the width of the page, and column names are escaped.
fn df_to_latex(df: &DataFrame) -> String {
    let mut latex = String::new();
    let num_cols = df.width();

    // Begin the tabularx environment with dynamic p-columns.
    // This uses TeX’s \dimexpr to compute each column’s width as:
    // \textwidth divided by num_cols minus 2\tabcolsep (for cell padding)
    write!(
        latex,
        "\\begin{{tabularx}}{{\\textwidth}}{{|*{num_cols}{{p{{\\dimexpr\\textwidth/{num_cols}-2\\tabcolsep\\relax}}|}}}}\n\\hline"
    )
    .unwrap();
    // Begin the tabularx environment. Each column is given as an 'X' (flexible column).
    // latex.push_str("\\begin{tabularx}{\\textwidth}{|");
    // for _ in 0..df.width() {
    //     latex.push_str("X|");
    // }
    // latex.push_str("}\n\\hline\n");

    // Header row: escape underscores in column names.
    for (i, series) in df.get_columns().iter().enumerate() {
        let col_name = escape_latex(series.name());
        write!(latex, "{}", col_name).unwrap();
        if i < df.width() - 1 {
            latex.push_str(" & ");
        } else {
            latex.push_str(" \\\\\n\\hline\n");
        }
    }

    // Body rows.
    for row in 0..df.height() {
        for (j, series) in df.get_columns().iter().enumerate() {
            let cell = series.get(row).unwrap();
            let cell_str = format_any_value(&cell);
            // Escape special characters in cell contents if needed.
            let cell_str = escape_latex(&cell_str);
            write!(latex, "{}", cell_str).unwrap();
            if j < df.width() - 1 {
                latex.push_str(" & ");
            } else {
                latex.push_str(" \\\\\n\\hline\n");
            }
        }
    }
    latex.push_str("\\end{tabularx}");
    latex
}

/// A newtype wrapper for Polars' DataFrame that implements a custom display for evcxr.
/// When output in an evcxr notebook, this prints LaTeX code between the special markers.
struct LaTeXDataFrame(DataFrame);

impl LaTeXDataFrame {
    pub fn evcxr_display(&self) {
        let latex = df_to_latex(&self.0);
        println!(
            "EVCXR_BEGIN_CONTENT text/latex\n{}\nEVCXR_END_CONTENT",
            latex
        );
    }
}

/// Formats an `AnyValue` to show just the underlying data,
/// rendering null values as "null".
fn format_any_value(value: &AnyValue) -> String {
    match value {
        AnyValue::Null => "null".to_owned(),

        AnyValue::String(s) => s.to_string(),

        AnyValue::StringOwned(s) => s.to_string(),

        AnyValue::Boolean(b) => b.to_string(),

        AnyValue::UInt32(u) => u.to_string(),

        AnyValue::UInt64(u) => u.to_string(),

        AnyValue::Int32(i) => i.to_string(),

        AnyValue::Int64(i) => i.to_string(),

        AnyValue::Float32(f) => {
            format!("{:.4}", f)
        }

        AnyValue::Float64(f) => {
            format!("{:.4}", f)
        }

        AnyValue::List(series) => {
            // Attempt to treat the list as a list of f64.
            if let Ok(ca) = series.f64() {
                let formatted: Vec<String> = ca
                    .into_iter()
                    .map(|opt| match opt {
                        Some(val) => {
                            format!("{:.4}", val)
                        }
                        None => "null".to_owned(),
                    })
                    .collect();
                format!("[{}]", formatted.join(", "))
            } else {
                // Fallback: if not a list of f64, use the default debug formatting.
                format!("{:?}", series)
            }
        }

        // Fallback for other types.
        _ => format!("{:?}", value),
    }
}

/// Converts a Polars DataFrame into an HTML table string.
/// The header includes the column name and its type,
/// while the cells display the unwrapped values, rendering nulls as "null".
fn df_to_html(df: &DataFrame) -> String {
    let mut html = String::new();

    // Start the table with a border.
    html.push_str("<table border=\"1\">");

    // Build header row with column names and types.
    html.push_str("<thead><tr>");

    for series in df.get_columns() {
        let col_name: &str = series.name();

        let dtype = series.dtype();

        // Map Polars data types to concise labels.
        let type_str: String = match dtype {
            DataType::String => "str".to_owned(),

            DataType::Float64 => "f64".to_owned(),

            DataType::Float32 => "f32".to_owned(),

            DataType::UInt32 => "u32".to_owned(),

            DataType::UInt64 => "u64".to_owned(),

            DataType::Int32 => "i32".to_owned(),

            DataType::Int64 => "i64".to_owned(),

            DataType::List(inner) => {
                if let DataType::Float64 = **inner {
                    "List<f64>".to_owned()
                } else if let DataType::Float32 = **inner {
                    "List<f32>".to_owned()
                } else {
                    format!("List<{:?}>", inner)
                }
            }

            _ => format!("{:?}", dtype),
        };

        write!(html, "<th>{} ({})</th>", col_name, type_str).unwrap();
    }

    html.push_str("</tr></thead>");

    // Build table body by iterating over rows.
    html.push_str("<tbody>");

    for i in 0..df.height() {
        html.push_str("<tr>");

        for series in df.get_columns() {
            // Get the value, which may be AnyValue::Null.
            let cell: AnyValue = series.get(i).unwrap();

            let cell_str = format_any_value(&cell);

            write!(html, "<td>{}</td>", cell_str).unwrap();
        }

        html.push_str("</tr>");
    }

    html.push_str("</tbody></table>");

    html
}

/// A newtype wrapper for Polars' DataFrame to enable custom evcxr display.
struct HTMLDataFrame(DataFrame);

impl HTMLDataFrame {
    /// Custom evcxr display method that outputs the HTML representation.
    pub fn evcxr_display(&self) {
        let html: String = df_to_html(&self.0);

        println!("EVCXR_BEGIN_CONTENT text/html\n{}\nEVCXR_END_CONTENT", html);
    }
}
