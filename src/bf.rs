#[derive(Clone, Debug)]
pub enum BFInstructions {
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

impl BFInstructions {
    pub fn from_char(x: char) -> Self {
        match x {
            '+' => BFInstructions::Plus,
            '-' => BFInstructions::Minus,
            '>' => BFInstructions::Left,
            '<' => BFInstructions::Right,
            '[' => BFInstructions::Open,
            ']' => BFInstructions::Close,
            ',' => BFInstructions::Input,
            '.' => BFInstructions::Output,
            _ => panic!(),
        }
    }

    pub fn from_str(x: &str) -> Vec<Self> {
        x.chars().map(BFInstructions::from_char).collect()
    }
}

#[derive(Clone, Debug)]
pub struct BFInterp {
    pub instructs: Vec<BFInstructions>,
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
                BFInstructions::Plus => self.array[self.index] += 1,
                BFInstructions::Minus => self.array[self.index] -= 1,

                BFInstructions::Left => {
                    self.index += 1;
                    if self.index == self.array.len() {
                        self.array.push(0);
                    }
                }
                BFInstructions::Right => self.index -= 1,

                BFInstructions::Open => {
                    if self.array[self.index] == 0 {
                        let mut count = 1;

                        while count > 0 {
                            self.instruct_pointer += 1;

                            match self
                                .instructs
                                .get(self.instruct_pointer)
                                .expect("misformed program")
                            {
                                BFInstructions::Plus
                                | BFInstructions::Minus
                                | BFInstructions::Left
                                | BFInstructions::Right
                                | BFInstructions::Input
                                | BFInstructions::Output
                                | BFInstructions::Label(_) => {}

                                BFInstructions::Open => count += 1,
                                BFInstructions::Close => count -= 1,
                            }
                        }
                    }
                }
                BFInstructions::Close => {
                    if self.array[self.index] != 0 {
                        let mut count = 1;

                        while count > 1 {
                            self.instruct_pointer -= 1;

                            match self
                                .instructs
                                .get(self.instruct_pointer)
                                .expect("malformed program")
                            {
                                BFInstructions::Plus
                                | BFInstructions::Minus
                                | BFInstructions::Left
                                | BFInstructions::Right
                                | BFInstructions::Input
                                | BFInstructions::Output
                                | BFInstructions::Label(_) => {}

                                BFInstructions::Open => count -= 1,
                                BFInstructions::Close => count += 1,
                            }
                        }
                    }
                }

                BFInstructions::Input => self.array[self.index] = self.input.remove(0),
                BFInstructions::Output => self.output.push(self.array[self.index]),
                BFInstructions::Label(x) => {
                    self.instruct_pointer += 1;
                    return Some(&x);
                }
            }

            self.instruct_pointer += 1;
        }

        None
    }

    pub fn exec(&mut self) {
        while let Some(_) = self.exec_until_label() {}
    }
}
