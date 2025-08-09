#[derive(Clone, Debug)]
pub enum BFInstruction {
    Plus,
    Minus,
    Left,
    Right,
    Open,
    Close,
    Input,
    Output,
    Label(String),
}

impl BFInstruction {
    pub fn from_char(x: char) -> Self {
        match x {
            '+' => BFInstruction::Plus,
            '-' => BFInstruction::Minus,
            '>' => BFInstruction::Left,
            '<' => BFInstruction::Right,
            '[' => BFInstruction::Open,
            ']' => BFInstruction::Close,
            ',' => BFInstruction::Input,
            '.' => BFInstruction::Output,
            _ => panic!(),
        }
    }

    pub fn from_str(x: &str) -> Vec<Self> {
        x.chars().map(BFInstruction::from_char).collect()
    }

    pub fn to_str(vec: &[BFInstruction]) -> String {
        vec.iter().map(|x| x.to_str_instruct()).collect()
    }

    pub fn to_str_instruct(&self) -> &str {
        match self {
            BFInstruction::Plus => "+",
            BFInstruction::Minus => "-",
            BFInstruction::Left => ">",
            BFInstruction::Right => "<",
            BFInstruction::Open => "[",
            BFInstruction::Close => "]",
            BFInstruction::Input => ",",
            BFInstruction::Output => ".",
            BFInstruction::Label(x) => &x,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BFInterp {
    pub instructs: Vec<BFInstruction>,
    pub instruct_pointer: usize,

    pub output: Vec<u8>,
    pub input: Vec<u8>,

    pub array: Vec<u8>,
    pub index: usize,
}

impl Default for BFInterp {
    fn default() -> Self {
        let mut array = Vec::new();
        array.push(0);

        BFInterp {
            array,
            instructs: Default::default(),
            instruct_pointer: Default::default(),
            output: Default::default(),
            input: Default::default(),
            index: Default::default(),
        }
    }
}

impl BFInterp {
    // will return label that is stopped at or None if its at the end of instructs
    pub fn exec_until_label(&mut self) -> Option<&str> {
        // dbg!(&self);

        while let Some(instruct) = self.instructs.get(self.instruct_pointer) {
            dbg!(&self);

            match instruct {
                BFInstruction::Plus => self.array[self.index] += 1,
                BFInstruction::Minus => self.array[self.index] -= 1,

                BFInstruction::Left => {
                    self.index += 1;
                    if self.index == self.array.len() {
                        self.array.push(0);
                    }
                }
                BFInstruction::Right => self.index -= 1,

                BFInstruction::Open => {
                    if self.array[self.index] == 0 {
                        let mut count = 1;

                        while count > 0 {
                            self.instruct_pointer += 1;

                            match self
                                .instructs
                                .get(self.instruct_pointer)
                                .expect("misformed program")
                            {
                                BFInstruction::Plus
                                | BFInstruction::Minus
                                | BFInstruction::Left
                                | BFInstruction::Right
                                | BFInstruction::Input
                                | BFInstruction::Output
                                | BFInstruction::Label(_) => {}

                                BFInstruction::Open => count += 1,
                                BFInstruction::Close => count -= 1,
                            }
                        }
                    }
                }
                BFInstruction::Close => {
                    if self.array[self.index] != 0 {
                        let mut count = 1;

                        while count > 0 {
                            self.instruct_pointer -= 1;

                            match self
                                .instructs
                                .get(self.instruct_pointer)
                                .expect("malformed program")
                            {
                                BFInstruction::Plus
                                | BFInstruction::Minus
                                | BFInstruction::Left
                                | BFInstruction::Right
                                | BFInstruction::Input
                                | BFInstruction::Output
                                | BFInstruction::Label(_) => {}

                                BFInstruction::Open => count -= 1,
                                BFInstruction::Close => count += 1,
                            }
                        }
                    }
                }

                BFInstruction::Input => self.array[self.index] = self.input.remove(0),
                BFInstruction::Output => self.output.push(self.array[self.index]),
                BFInstruction::Label(x) => {
                    self.instruct_pointer += 1;
                    return Some(&x);
                }
            }

            self.instruct_pointer += 1;
        }

        None
    }

    pub fn _exec(&mut self) {
        while let Some(_) = self.exec_until_label() {}
    }
}
