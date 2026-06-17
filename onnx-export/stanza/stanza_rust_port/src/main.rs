mod model;
use tch::{nn, Device};
use model::pos::Tagger;

fn main() {
    let vs = nn::VarStore::new(Device::Cpu);
    let _tagger = Tagger::new(&vs.root(), 1000, 1000, 200, 5);
    println!("✅ Tagger 구조체 초기화 성공");
}
