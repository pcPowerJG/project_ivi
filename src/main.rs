
extern crate EvaTwo;
use EvaTwo::deep_learning::*;
use std::fs::File;
use std::io::prelude::*;
use std::str::FromStr;

fn out_layer_error(true_res: Vec<f32>, fact_res: Vec<f32>) -> Vec<f32> {
    let mut vec_error: Vec<f32> = Vec::new();
    for i in 0..fact_res.len() {
        vec_error.push(//0.5 * 
           /* (*/(true_res[i] - fact_res[i])/* * (true_res - fact_res))*/
        );
    }
    Vec::new()
}

fn answer_function(a: f32) -> f32 { if a > 0.0 { a } else { 0.0 } }
fn bar (a: f32) -> Vec<f32> { Vec::new() }
fn get_targets(indx: usize) -> Vec<f32> {
    let mut targets: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    for i in 0..10 {
        if i == indx {
            targets[i] = 0.99;
        } else {
            targets[i] = 0.01;
        }
    } targets
}
fn get_inputs(data: Vec<&str>) -> Vec<f32> {
    let mut inputs: Vec<f32> = Vec::new();
    for _ in 0..784 { inputs.push(0.0); }
    for i in 1..785 {
        inputs[i-1] = (f32::from_str(data[i]).unwrap() / 255.0 * 0.99) + 0.01;
    } inputs
}


fn main(){
    //fn new(inputs_: Vec<usize>, outputs_: Vec<usize>) -> Layer {
    let mut layer = Layer::new(
        vec![784, 200], 
        vec![200, 10],
        vec![0.5, 0.5]
    );
    //layer.save_to_file("file.csb");
    //layer.load_from_file("file.csb");
    //layer.println();    
    //panic!("");
    let mut file = match File::open("mnist_test.csv") {
        Ok(A) => A,
        Err(e) => { println!("{:?}", e); panic!(""); File::open("other_file.cvs").unwrap() }
    };
    let mut content = String::new();
    file.read_to_string(&mut content).expect("не удалсь прочитать файл");
    let data_set: Vec<&str> = content.split('\n').collect();
    //println!("content_line_count: {}", data_set.len());
    //panic!("stop");
    println!("data_set count: {}", data_set.len());
    println!("начали обучение");
    for epoh in 0..10 {
        let mut _u: usize = 0;
        println!("epoh: [{}]", epoh);
        for number in &data_set {
            if _u % 100 == 0 && _u != 0{
                println!("{}", _u);
                //break;
            }
            //if _u % 300 == 0 && _u != 0 { break; }
            let data: Vec<&str> = number.split(',').collect();
            let mut target: usize = 0;
            if data[0] != "" || data[0] != "\0" {
                target = match usize::from_str(data[0]) {
                    Ok(A) => A,
                    Err(e) => { 
                        println!("error to convertation word [{}] in iteration [{}]", number.clone(), _u.clone()); 
                        continue;
                        0
                    }
                };
            }              
            let targets: Vec<f32> = get_targets(target);
            let inputs: Vec<f32> = get_inputs(data.clone());
            layer.train(inputs, targets);
            _u += 1;
        }
    
        //panic!("");

        let mut file = match File::open("mnist_train.csv") {
            Ok(A) => A,
            Err(e) => { println!("{:?}", e); panic!(""); File::open("other_file.cvs").unwrap() }
        };
        let mut content: String = String::new();
        match file.read_to_string(&mut content) {
            Ok(A) => {},
            Err(e) => { panic!("не удалось прочитать файл, ошибка: {:?}", e); }
        }
        let data_set: Vec<&str> = content.split('\n').collect();
        let mut tr_ans: usize = 0;
        let mut i: usize = 0;
        for number in &data_set {        
            let data: Vec<&str> = number.split(',').collect();
            let mut target: usize = 0;
            let data: Vec<&str> = number.split(',').collect();
            let mut target: usize = 0;
            if data[0] != "" || data[0] != "\0" {
                target = match usize::from_str(data[0]) {
                    Ok(A) => A,
                    Err(e) => { 
                        println!("error to convertation test word [{}] in iteration [{}]", number.clone(), i.clone()); 
                        continue;
                        0
                    }
                };
            } 
            let targets: Vec<f32> = get_targets(target);
            let inputs: Vec<f32> = get_inputs(data.clone());
            if layer.query(inputs) == target {
                tr_ans += 1;
            }
            //println!("\ttrue result: [{}]", data[0].clone());
            //if i % 20 == 0 && i != 0 { break; }
            i += 1;
        }
        println!(
            "правильных результатов на тестовый выборке в эпохе [{}]: {}", epoh, tr_ans
        );
        let mut save: String = "file".to_string();
        save += epoh.to_string().as_str();
        save += ".smb";
       match layer.save_to_file(save.as_str()) {
           Ok(A) => {},
           Err(e) => panic!(""), 
       }
    }
    let picture: Vec<f32> = vec![
            0.0, 0.1, 0.2, 0.3,
            0.4, 0.5, 0.6, 0.7,
            0.8, 0.9, 0.10, 0.11,
            0.12, 0.13, 0.14, 0.15
        ];
    let mut matrix: Vec<f32> = vec![
        1.0, 1.0,
        1.0, 1.0
    ];
    /*
     // радиус матрицы (ширина), матрица, позицияХ, позицияY, изображение, ширина изображения, до конца?
        pub fn convolutional(&mut self, 
            rectagle_radius: usize, 
            matrix: &Vec<f32>, 
            mut positionX: usize, 
            mut positionY: usize, 
            picture: &Vec<f32>, 
            picture_wight: usize
        )
        */        
        //println!("new picture[\n {:?}\n]", a1.convolutional(2, &matrix, 0, 0, &picture, 4));
}