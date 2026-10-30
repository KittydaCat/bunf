
pub struct VecWrapper(Vec<usize>);

impl VecWrapper {
    
    pub fn push(&mut self, x: usize) {self.0.push(x)}
    pub fn pop(&mut self) -> usize {self.0.pop().unwrap()}

    pub fn index(&mut self, x: usize) -> usize {*self.0.get(x).unwrap()}
    pub fn index_set(&mut self, x: usize, y: usize) {self.0[x] = y}


}

pub fn list() -> VecWrapper {
    VecWrapper(Vec::new())
}

pub fn output(c: usize) {
    print!("{}", c as u8 as char)
}

pub fn input() -> usize {
    todo!()
}

pub fn panic() {panic!()}
