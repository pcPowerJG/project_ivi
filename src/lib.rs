
pub mod deep_learning {
    use std::{
        future::Future,
        io::{self, Read, Write},
        net::{TcpListener, TcpStream},
        pin::Pin,
        task::{Context, Poll},
        thread::{self, JoinHandle},
        fs::File,
        io::prelude::*,
        str::FromStr,
    };
    use rand::prelude::*;
    
    pub struct vis_data{
        pixel: Vec<u8>,
        ip_addr: String
    }
    pub struct mind_object {
        data: Vec<u8>,
        name: String,
    }
    impl vis_data {
        pub fn new(pxl: Vec<u8>, adrs: String) -> vis_data {
            vis_data { pixel: pxl, ip_addr: adrs }
        }
        pub fn what_is_that(&self) -> mind_object {
           
           return mind_object { data: Vec::new(), name: String::new() }
        }
    }

    pub struct Neyron {
		in_: usize,
        out_: usize,
        matrix: Vec<Vec<f32>>,
        hidden: Vec<f32>,
        errors: Vec<f32>,
        learn_rate: f32       
		//activation_function: Fn(Vec<f32>) -> f32,
        /*
            fn call_with_one<F>(func: F) -> usize
                where F: Fn(usize) -> usize {
                func(1)
            }
            let double = |x| x * 2;
            assert_eq!(call_with_one(double), 2);
        */
		//(answer_function: F) where F: Fn(f32, usize) -> f32 
	}
    pub struct Layer {
        neyrons: Vec<Neyron>,
        _input_neurons: usize,
        _output_neurons: usize,
        _nl_count: usize,
        _inputs: Vec<f32>,
        _targets: Vec<f32>
        /* std::vector<nnLay> *_nList = nullptr;
        int _inputNeurons;
        int _outputNeurons;
        int _nlCount;

        float *_inputs = nullptr;
        float *_targets = nullptr;*/
    }
    impl Layer {        
        pub fn new(inputs_: Vec<usize>, outputs_: Vec<usize>, learn_rate: Vec<f32>) -> Layer {            
            let mut nyrs: Vec<Neyron> = Vec::new();
            for i in 0..inputs_.len() {
                let mut neyron = Neyron::new(inputs_[i], outputs_[i], learn_rate[i]);
                neyron.set_IO(inputs_[i], outputs_[i]);
                nyrs.push(neyron);
            } 
            Layer { 
                neyrons: nyrs, 
                _input_neurons: inputs_[0], 
                _output_neurons: outputs_[outputs_.len() - 1].clone(), 
                _nl_count: inputs_.len(),
                _inputs: Vec::new(),
                _targets: Vec::new()
            }
        }
        fn feed_forwarding(&mut self, ok: bool) -> usize {
            //--- signal through NN in forward direction

            //--- for first layer argument is _inputs
            self.neyrons[0].make_hidden(self._inputs.clone());
        //--- for other layer argument is "hidden" array previous's layer
            for i in 1..self.neyrons.len() {
                let hidden = self.neyrons[i-1].get_hidden();
                self.neyrons[i].make_hidden(hidden);
            }


            //--- bool condition for query NN or train NN
            if !ok {
                //println!("Feed Forward: ");
                let mut max: f32 = 0.0;
                let mut mx: usize = 0;
                for out in 0..self._output_neurons {
                    //println!("[{}],: {}", out, self.neyrons[self._nl_count-1].hidden[out]);
                    if self.neyrons[self._nl_count-1].hidden[out] > max {
                        max = self.neyrons[self._nl_count-1].hidden[out];
                        mx = out;
                    }
                    //println!("matrix: {:?}", self.neyrons[self._nl_count-1].matrix);
                }
                //println!("<<[{}]", mx);
                return mx;
            } else {
                // printArray(list[3].getErrors(),list[3].getOutCount());
                self.back_propagate(0.5);
            } 0
        }
        fn back_propagate(&mut self, learn_rate: f32) {   
            //--- calculate errors for last layer
            //println!("tar: {:?}", self._targets.clone());
            self.neyrons[self._nl_count-1].calc_out_error(self._targets.clone());
            //--- for others layers to calculate errors we need information about "next layer"
            //---   //for example// to calculate 4'th layer errors we need 5'th layer errors
            let mut k: usize = self._nl_count - 2;
            loop {
                let err = self.neyrons[k + 1].get_errors(); 
                //println!("err: {:?}", err.clone());
                let mtrx = self.neyrons[k + 1].get_matrix();
                let incnt = self.neyrons[k + 1].get_in_count();
                let otct = self.neyrons[k + 1].get_out_count();
                self.neyrons[k].calc_hid_error(
                    err,
                    mtrx,
                    incnt,
                    otct
                );
                if k == 0 { break; }
                k -= 1;
            }
            /*
                for (int i = _nlCount-2; i>=0; i--)
                    _nList->at(i).calcHidError(
                        _nList->at(i+1).getErrors(),
                        _nList->at(i+1).getMatrix(),
                        _nList->at(i+1).getInCount(),
                        _nList->at(i+1).getOutCount()
                    );
            */
            //--- updating weights
            //--- to UPD weight for current layer we must get "hidden" value array of previous layer
            k = self._nl_count - 1;
            loop {
                if !(k > 0) { break; }
                // entered_value: Vec<f32>, learn_rate: f32
                let hidden = self.neyrons[k - 1].get_hidden();
                self.neyrons[k].matrix_update(hidden, learn_rate.clone());
                k -= 1;
            }
            /*for (int i = _nlCount-1; i>0; i--)
                _nList->at(i).updMatrix(_nList->at(i-1).getHidden());*/
            //--- first layer hasn't previous layer.
            //--- for him "hidden" value array of previous layer be NN input
            self.neyrons[k].matrix_update(self._inputs.clone(), learn_rate.clone());
        }
        pub fn train(&mut self, in_: Vec<f32>, out_: Vec<f32>) {
            if (in_.len() > 0) {
                self._inputs = in_.clone();
            }
            if out_.len() > 0 {
                self._targets = out_.clone();
            }
            self.feed_forwarding(true);
            /*for i in 0..self.neyrons.len(){
                self.neyrons[i].println();
            }
            println!("\n\n\n");*/
        }
        pub fn query(&mut self, in_: Vec<f32>) -> usize {
            self._inputs = in_.clone();
            self.feed_forwarding(false)
        }
        // радиус матрицы (ширина), матрица, позицияХ, позицияY, изображение, ширина изображения, до конца?
        pub fn convolutional_extern_matrix(&self, 
            rectagle_radius: usize, matrix: &Vec<f32>, 
            mut positionX: usize, mut positionY: usize, 
            picture: &Vec<f32>, picture_wight: usize
        ) -> Vec<f32> {            
            let mut matrix_coordinate: Vec<usize> = Vec::new();
            let hight: usize = picture.len() / picture_wight.clone();                      
            let mut con_vector: Vec<f32> = Vec::new();            
            for h in 0..(hight-rectagle_radius + 1) {
                for w in 0..(picture_wight-rectagle_radius+1) {
                    for y in 0..rectagle_radius {                    
                            for x in 0..rectagle_radius {
                                //println!("into for: pos x - {}, pos y - {}", positionX, positionY);
                                matrix_coordinate.push(
                                    positionY + positionX
                                );                                
                                con_vector.push(matrix[x + y * rectagle_radius] * picture[positionX + positionY]);
                                positionX += 1;                            
                            }
                        positionX -= rectagle_radius;
                        positionY += picture_wight;                    
                    }
                    positionY -= picture_wight * rectagle_radius;
                    //println!(" after for:\n pos x - {}, pos y - {}", positionX, positionY);                    
                    positionX +=1;                
                }
                positionX = 0;
                positionY += hight;
            }            
            //println!("\n{:?}", matrix_coordinate.clone());
            //println!("*****pixels*filters*****\n{:?}", con_vector);
            let mut new_picture: Vec<f32> = Vec::new();
            let mut summ: f32 = 0.0;
            for index in 0..con_vector.len() {                
                if (index % (rectagle_radius * rectagle_radius) == 0) && (index != 0) {
                    new_picture.push(summ);
                    summ = 0.0;
                } 
                summ += con_vector[index].clone();
            } new_picture.push(summ);
            //println!("new picture:\nlen: {}\npixel: {:?}", new_picture.len(), new_picture);
            new_picture
        }

        pub fn convolutional(&self, 
            rectagle_radius: usize,
            mut positionX: usize, mut positionY: usize, 
            picture: &Vec<f32>, picture_wight: usize
        ) -> Vec<f32> {            
            let mut matrix: Vec<f32> = Vec::new();
            self.neyrons[0].matrix_to_one_string_array(&mut matrix);
            let mut matrix_coordinate: Vec<usize> = Vec::new();
            let hight: usize = picture.len() / picture_wight.clone();                      
            let mut con_vector: Vec<f32> = Vec::new();            
            for h in 0..(hight-rectagle_radius + 1) {
                for w in 0..(picture_wight-rectagle_radius+1) {
                    for y in 0..rectagle_radius {                    
                            for x in 0..rectagle_radius {
                                //println!("into for: pos x - {}, pos y - {}", positionX, positionY);
                                matrix_coordinate.push(
                                    positionY + positionX
                                );                                
                                con_vector.push(matrix[x + y * rectagle_radius] * picture[positionX + positionY]);
                                positionX += 1;                            
                            }
                        positionX -= rectagle_radius;
                        positionY += picture_wight;                    
                    }
                    positionY -= picture_wight * rectagle_radius;
                    //println!(" after for:\n pos x - {}, pos y - {}", positionX, positionY);                    
                    positionX +=1;                
                }
                positionX = 0;
                positionY += hight;
            }            
            //println!("\n{:?}", matrix_coordinate.clone());
            //println!("*****pixels*filters*****\n{:?}", con_vector);
            let mut new_picture: Vec<f32> = Vec::new();
            let mut summ: f32 = 0.0;
            for index in 0..con_vector.len() {                
                if (index % (rectagle_radius * rectagle_radius) == 0) && (index != 0) {
                    new_picture.push(summ);
                    summ = 0.0;
                } 
                summ += con_vector[index].clone();
            } new_picture.push(summ);
            //println!("new picture:\nlen: {}\npixel: {:?}", new_picture.len(), new_picture);
            new_picture
        }
        // FILE SYSTEM
        /* */
        pub fn save_to_file<'a>(&self, file_name: &'a str) -> Result<bool, &'a str> {
            // NEYRON: 
            //pub fn save_to_string(&self) -> String {
                // output: this neyron to text_format + '_' symbol            
            let mut save_string: String = String::new();            
            for i in 0..self.neyrons.len() {
                save_string += self.neyrons[i].save_to_string().as_str();
                save_string.push('_');
            }            
            let len_: usize = save_string.len();
            let save_string: String = save_string.split_at(len_ - 1).0.to_string();
            let mut file: File = match File::create(file_name) {
                Ok(A) => A,
                Err(e) => { 
                    println!("{:?}", e); 
                    drop(save_string); 
                    return Err("error_create_file"); 
                    File::create(file_name).unwrap() 
                },
            };
            match file.write(save_string.as_bytes()) {
                Ok(A) => {},
                Err(e) => { println!("{:?}", e); return Err("error_write_file"); },
            }
            Ok(true)
        }
        pub fn load_from_file<'a>(&mut self, file_path: &'a str) -> Result<bool, &'a str> {
            //pub fn import_from_string(&mut self, sting_load: String) {
            let mut content: String = String::new();
            let mut file: File = match File::open(file_path) {
                Ok(A) => A,
                Err(e) => { 
                    println!("{:?}", e); 
                    drop(content);
                    return Err("error_open_file"); 
                    File::open(file_path).unwrap() 
                },
            };
            match file.read_to_string(&mut content) {
                Ok(A) => {},
                Err(e) => {
                    println!("{:?}", e); 
                    drop(content);
                    return Err("error_read_file");
                },
            }
            Ok(true)
        }
        pub fn println(&self) {
            for i in 0..self.neyrons.len() {
                self.neyrons[i].println();
                println!("------------------------");
            }
        }
        pub fn set_learn_rate_to_vec(&mut self, new_rate: Vec<f32>) -> Result<bool, &str> {
            if new_rate.len() != self.neyrons.len() {
                return Err("new_rate.len");
            }
            for i in 0..new_rate.len() {
                self.neyrons[i].set_learn_rate(new_rate[i].clone());
            }
            Ok(true)
        }
        pub fn set_learn_rate_at(&mut self, index: usize, learn_rate: f32) -> Result<bool, &str> {
            if index >= self.neyrons.len() {
                Err("index > fact_len")
            } else {
                self.neyrons[index].set_learn_rate(learn_rate);
                Ok(true)
            }
        }
    }
    impl Neyron{
        pub fn get_errors(&self) -> Vec<f32> { self.errors.clone() }
        pub fn get_matrix(&self) -> Vec<Vec<f32>> { self.matrix.clone() }
        pub fn matrix_to_one_string_array(&self, vector_: &mut Vec<f32>) {
            for _i in 0..(self.in_+1) {                
                for _j in 0..self.out_ {                    
                    vector_.push(self.matrix[_i][_j]);
                }
            }
        }
        pub fn get_in_count(&self) -> usize { self.in_.clone() }
        pub fn get_out_count(&self) -> usize { self.out_.clone() }
        pub fn set_learn_rate(&mut self, learn_rate: f32) {
            self.learn_rate = learn_rate;
        }        
        pub fn new(input: usize, output: usize, learn_rate: f32) -> Neyron {
            Neyron { 
                in_: input,
                out_: output,
                matrix: Vec::new(),
                hidden: Vec::new(),
                errors: Vec::new(),
                learn_rate: learn_rate,
            }
        }
        pub fn get_hidden(&self) -> Vec<f32> { self.hidden.clone() }
        pub fn save_to_string(&self) -> String {
            let mut save_string: String = String::new();
            /*
                in_: usize,
                out_: usize,
                matrix: Vec<Vec<f32>>,
                hidden: Vec<f32>,
                errors: Vec<f32>
            */
            save_string.push('[');
            save_string += self.in_.to_string().as_str();
            save_string.push('|');
            save_string += self.out_.to_string().as_str();
            save_string.push('|');
            save_string += self.learn_rate.to_string().as_str();
            save_string.push(']');
            save_string.push('{');
            for _i in 0..(self.in_+1) {
                //let mut _matrix: Vec<f32> = Vec::new();
                save_string.push('r');
                for _j in 0..self.out_ {                    
                    save_string.push('c');
                    save_string += self.matrix[_i][_j].to_string().as_str();                    
                }
            }
            save_string.push('}');            
            save_string
        }
        pub fn import_from_string(&mut self, string_load: String) {
            let input_count: usize = match usize::from_str(string_load.clone().as_str()
                .split('[').collect::<Vec<&str>>()[1].split('|').collect::<Vec<&str>>()[0]){
                    Ok(A) => A,
                    Err(e) => { panic!("{:?}", e); 0 },
            };
            let output_count: usize = match usize::from_str(string_load.clone().as_str()
                .split('|').collect::<Vec<&str>>()[1]){
                    Ok(A) => A,
                    Err(e) => { panic!("{:?}", e); 0 },
            };
            let learn_rate: f32 = match f32::from_str(string_load.clone().as_str()
                .split('|').collect::<Vec<&str>>()[2]
                    .split(']').collect::<Vec<&str>>()[0]) {
                        Ok(A) => A,
                        Err(e) => { panic!("{:?}", e); 0.0 }
            };
            let string_load: String = string_load.clone().as_str().split('{').collect::<Vec<&str>>()[1]
                .split('}').collect::<Vec<&str>>()[0].to_string();
            let rows_: Vec<&str> = string_load.split('r').collect::<Vec<&str>>()[1..].to_vec();
            for row in rows_ {
                let mut mtrx: Vec<f32> = Vec::new();
                let collums: Vec<&str> = row.split('c').collect::<Vec<&str>>()[1..].to_vec();
                //self.matrix.push(_matrix.clone());
                for col in collums {
                    mtrx.push(
                        match f32::from_str(col) {
                            Ok(A) => A,
                            Err(e) => {
                                panic!("{:?}", e); 0.0
                            }
                        }
                    );
                } 
                self.learn_rate = learn_rate;
                self.matrix.push(mtrx);
            }            
        }
        pub fn println(&self) {
            println!("in_: {}\nout_:{}\nlearn_rate: {}\nmatrix:{:?}\nhidden:{:?}\nerrors:{:?}", self.in_,
            self.out_, self.learn_rate, self.matrix, self.hidden, self.errors);
        }
        pub fn set_IO(&mut self, inputs: usize, outputs: usize) {
            //--- initialization values and allocating memory
            self.in_=inputs;
            self.out_=outputs;
            self.errors = Vec::new();
            for _ in 0..outputs {
                self.errors.push(0.0);
            }
            self.hidden = Vec::new();
            for _ in 0..outputs {
                self.hidden.push(0.0);
            }
            self.matrix = Vec::new();
            let mut rng = rand::thread_rng();
            for _i in 0..(inputs+1) {
                let mut _matrix: Vec<f32> = Vec::new();
                for _j in 0..outputs {                    
                    let y: f32 = rng.gen(); // generates a float between 0 and 1                    
                    _matrix.push(y.clone()/1000.0);
                }
                self.matrix.push(_matrix.clone());
            }
            drop(rng);            
        }
        fn sigmoid(x: f32) -> f32 {
            let mut y: f32 = 0.0;
            y = 1.0 / ((-1.0 * x).exp() + 1.0);
            //println!("sigmoid -> x: {}, y: {}", x, y);
            y
        }
        pub fn correct_weight<F>(&mut self, numeric_error_value: f32, type_err: &'static str, answer_function: F, a: f32) -> Vec<f32>
        where F: Fn(f32) -> Vec<f32> {
            let mut err_vec: Vec<f32> = Vec::new();
            //println!("веса: {:?}", self.weight.clone());
            /*for i in 0..self.weight.clone().len(){
                //err_vec.push(numeric_error_value * self.weight[i]);
                //self.weight[i] *= numeric_error_value;
            }*/
            err_vec
        }      

        pub fn matrix_update(&mut self, entered_value: Vec<f32>, learn_rate: f32) {
            for ou in 0..self.out_ {
                for hid in 0..self.in_ {
                    self.matrix[hid][ou] += (learn_rate * self.errors[ou] * entered_value[hid]);
                }
                self.matrix[self.in_][ou] += (learn_rate * self.errors[ou]);
            }
        }
        //fn set_io(inputs:)
        pub fn make_hidden(&mut self, inputs: Vec<f32>) {
            for hid in 0..self.out_ {
                let mut tmp_s: f32 = 0.0;
                for inp in 0..self.in_ {
                    tmp_s += inputs[inp] * self.matrix[inp][hid];
                }
                tmp_s += self.matrix[self.in_][hid];
                //println!("tmp_s: {}", tmp_s);
                self.hidden[hid] = Neyron::sigmoid(tmp_s);
            }
        }
        pub fn sigmoid_as_derivate(&self, value: f32) -> f32 { value * (1.0 - value) }
        pub fn calc_out_error(&mut self, targets: Vec<f32>) {            
            for ou in 0..self.out_ {
                self.errors[ou] = (targets[ou] - self.hidden[ou]) * self.sigmoid_as_derivate(self.hidden[ou]);
            }
        }
        pub fn calc_hid_error(&mut self, targets: Vec<f32>, out_weights: Vec<Vec<f32>>, in_s: usize, out_s: usize) {
            for hid in 0..in_s {
                self.errors[hid] = 0.0;
                for ou in 0..out_s {
                    self.errors[hid] += targets[ou] * out_weights[hid][ou];
                }
                self.errors[hid] *= self.sigmoid_as_derivate(self.hidden[hid]);
            }
        }
    }
}