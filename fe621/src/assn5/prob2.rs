use burn::backend::Wgpu;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::DataframeDataset;
use burn::nn::conv::Conv2d;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch_size, height] = features.dims();

        // Create a channel at the second dimension.
        let x = features.reshape([batch_size, 16 * 8 * 8]);

        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            activation: Relu::new(),
        }
    }
}

fn data_load() -> Result<DataFrame, PolarsError> {
    // Path to the data directory
    let data_dir = "data";

    // Get all CSV file paths in the directory
    let csv_files = fs::read_dir(data_dir)
        .expect("Failed to read data directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension()? == "csv" {
                Some(path)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let regex = regex::Regex::new(r"(calls|puts)_(\d{4})-(\d{2})-(\d{2})\.csv").unwrap();
    // Load each CSV file into a LazyFrame
    let mut lazy_frames = Vec::new();
    for file in csv_files {
        let mut lf = LazyCsvReader::new(file.to_str().unwrap())
            .with_has_header(true)
            .finish()?;
        let file_name = file.file_name().unwrap().to_str().unwrap();
        let captures = regex.captures(file_name).unwrap();
        let is_call = &captures[1] == "calls";
        let year: i32 = captures[2].parse().unwrap();
        let month: u32 = captures[3].parse().unwrap();
        let day: u32 = captures[4].parse().unwrap();

        let time_to_maturity = ((datetime(DatetimeArgs::new(lit(year), lit(month), lit(day)))
            .dt()
            .date()
            - datetime(DatetimeArgs::new(lit(2025), lit(5), lit(8)))
                .dt()
                .date())
        .cast(DataType::Float32)
            / lit(1000.0 * 60.0 * 60.0 * 24.0 * 365.0))
        .alias("T");

        let asset_price = lit(567.77).alias("S").cast(DataType::Float32);
        let strike = col("strike").alias("K").cast(DataType::Float32);
        let is_call = lit(is_call).alias("is_call");
        let bid = col("bid").cast(DataType::Float32);
        let ask = col("ask").cast(DataType::Float32);
        lf = lf.select(&[asset_price, strike, time_to_maturity, is_call, bid, ask]);
        lazy_frames.push(lf);
    }

    // Concatenate all LazyFrames into a single LazyFrame
    let combined_lazy_frame = concat(lazy_frames, UnionArgs::default())?;

    // Collect the LazyFrame into a DataFrame
    let df = combined_lazy_frame.collect()?;
    Ok(df)
}

#[derive(Clone, Debug, Default)]
struct DataBatcher;

struct Data<B: Backend> {
    features: Tensor<B, 2>,
    labels: Tensor<B, 2>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DataItem {
    // features
    pub S: f32,
    pub K: f32,
    pub T: f32,
    pub is_call: bool,

    // labels
    pub bid: f32,
    pub ask: f32,
}

impl<B: Backend> Batcher<B, MnistItem, Data<B>> for DataBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // Normalize: scale between [0,1] and make the mean=0 and std=1
            // values mean=0.1307,std=0.3081 are from the PyTorch MNIST example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([(item.label as i64).elem::<B::IntElem>()], device)
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        DataBatch { images, targets }
    }
}

pub fn a() {
    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();
    let model = ModelConfig::new(10, 512).init::<MyBackend>(&device);

    let batcher = DataBatcher::default();

    let df = data_load().unwrap();
    println!("{}", df);

    let dataset: DataframeDataset<MyBackend> = DataframeDataset::new(df).unwrap();
    let dataloader = DataLoaderBuilder::new(batcher.clone());
}
