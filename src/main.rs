mod bf;

use crate::bf::{BFInstructions, BFInterp};

#[derive(Debug, Clone)]
pub enum PrimativeType {
    Int(u8),
    Boolean(bool),
    // List(Vec<PrimativeType>),
    Uninit,
}

impl PrimativeType {
    fn traverse(&self) -> (&'static str, &'static str) {
        match self {
            PrimativeType::Int(_) | PrimativeType::Boolean(_) | PrimativeType::Uninit => (">", "<"),
        }
    }

    // this is a butt done of heap allocs might be slow as fuck
    fn to_bf(&self) -> Vec<u8> {
        match self {
            PrimativeType::Int(x) => vec![*x],
            PrimativeType::Boolean(x) => vec![*x as u8],
            PrimativeType::Uninit => vec![0],
        }
    }

    fn bf_len(&self) -> usize {
        match self {
            PrimativeType::Int(_) => 1,
            PrimativeType::Boolean(_) => 1,
            PrimativeType::Uninit => 1,
        }
    }
}

// #[derive(Debug, Clone)]
// pub enum BFAsmTypes<T: Value> {
//     Int(T),
//     Boolean(T),
//     List(Vec<T>),
//     Uninit,
// }
//
// trait Value {
//     // const IS_MOD: bool;
//     fn set(&mut self, value: u8);
// }

// basic Instructions
// the initial usize is the the stack index of the value you want to do the op on
// the args are usually passed positionally i.e. Add will check that there are 2 ints at the usize
// it is given the MoveVal and Control flow ops are the notable exceptions,
struct BFAsmInstruction {
    target: usize,
    variant: InstructionVariant,
}

impl BFAsmInstruction {
    fn new(target: usize, variant: InstructionVariant) -> Self {
        BFAsmInstruction { target, variant }
    }
}
pub enum InstructionVariant {
    // bf ptr ops
    // MovePtrTo(usize),

    // Int ops
    IntSet(u8),
    IntAdd,
    IntConstAdd(u8),
    // IntDiff(usize),
    IntCopy,
    IntRemove,
    // IntSubCons(usize),
    // Cmp(usize), // how do we want this to work?
    // Match(usize, Vec<(u8, Vec<Instructions>)>),
    // Eq(usize),

    // Boolean Ops
    BoolSet(bool),
    BoolNot,
    BoolCopy,
    BoolRemove,
    // If(usize, Vec<Instructions>),
    While(Vec<BFAsmInstruction>),

    // List ops
    // ListInit(usize),
    // ListIndex(usize),
    // ListLen(usize),
    // ListPop(usize),
    // ListPush(usize),

    // IO
    Input,
    Output,
}

impl BFAsmInstruction {
    fn exec(&self, compiler: &mut impl BFAsmComplier) -> Result<(), CompilerError> {
        let target = self.target;
        match &self.variant {
            InstructionVariant::IntSet(val) => int_set(compiler, target, *val),
            InstructionVariant::IntAdd => int_add(compiler, target),
            InstructionVariant::IntConstAdd(val) => int_const_add(compiler, target, *val),
            InstructionVariant::IntCopy => int_copy(compiler, target),
            InstructionVariant::IntRemove => int_remove(compiler, target),
            InstructionVariant::BoolSet(val) => bool_set(compiler, target, *val),
            InstructionVariant::BoolNot => todo!(),
            InstructionVariant::BoolCopy => bool_copy(compiler, target),
            InstructionVariant::BoolRemove => bool_remove(compiler, target),
            InstructionVariant::While(_instructions) => todo!(),
            InstructionVariant::Input => input(compiler, target),
            InstructionVariant::Output => output(compiler, target),
        }
    }
}

#[derive(Clone, Debug)]
enum CompilerError {
    TypeMismatch,
}

trait BFAsmComplier {
    // array methods
    fn get_mut(&mut self, pos: usize, len: usize) -> &mut [PrimativeType];

    fn get_mut_chunk<const N: usize>(&mut self, pos: usize) -> &mut [PrimativeType; N] {
        self.get_mut(pos, N).first_chunk_mut().unwrap()
    }

    fn index(&mut self) -> &mut usize;

    // TODO Im not sure how I want to do the IO
    fn get_input(&mut self) -> u8;

    fn push_output(&mut self, val: u8);

    // writer methods
    fn write(&mut self, code: &str);

    fn label(&mut self, name: &str);

    fn enabled(&mut self) -> &mut bool;

    // fn exec(&mut self, x: &BFAsmInstruction) -> Result<(), CompilerError>;
}

// type BFArgs<'a> = (&'a mut dyn BFArray, &'a Option<&'a mut dyn BFWrite>, usize);
#[derive(Clone, Debug, Default)]
struct BFAsmArray {
    array: Vec<PrimativeType>,
    index: usize,
    input: Vec<u8>,
    output: Vec<u8>,
}

impl BFAsmArray {
    fn trim_back(&mut self) {
        while let Some(PrimativeType::Uninit) = self.array.last() {
            self.array.pop();
        }
    }

    fn get_mut(&mut self, pos: usize, len: usize) -> &mut [PrimativeType] {
        while pos + len > self.array.len() {
            self.array.push(PrimativeType::Uninit);
        }

        self.array.get_mut(pos..(pos + len)).unwrap()
    }

    fn get_mut_chunk<const N: usize>(&mut self, pos: usize) -> &mut [PrimativeType; N] {
        self.get_mut(pos, N).first_chunk_mut().unwrap()
    }

    fn get_input(&mut self) -> u8 {
        self.input.remove(0)
    }

    fn push_input(&mut self, val: u8) {
        self.input.push(val);
    }

    fn push_output(&mut self, val: u8) {
        self.output.push(val);
    }
}

#[derive(Clone, Debug)]
struct TestComplier {
    interp: bf::BFInterp,
    enabled: bool,
    array: BFAsmArray,
}

impl TestComplier {
    fn exec_instructs(&mut self, instructs: &[BFAsmInstruction]) {
        instructs.iter().for_each(|x| x.exec(self).unwrap())
    }

    fn push_input(&mut self, val: u8) {
        self.array.input.push(val);
        self.interp.input.push(val);
    }

    fn compare_states(&mut self) -> bool {
        dbg!("comparing states");
        dbg!(&self);

        if self.array.input != self.interp.input {
            dbg!("1");
            return false;
        }

        if self.array.output != self.interp.output {
            dbg!("2");
            return false;
        }

        // off by one???
        let array_pt_pos: usize = self
            .array
            .get_mut(0, self.array.index)
            .iter()
            .map(|x| x.bf_len())
            .sum();

        if dbg!(array_pt_pos) != self.interp.index {
            dbg!("3");
            return false;
        }

        self.array.trim_back();

        let mut iter = self.array.array.iter();

        let mut bf_index = 0;

        while let Some(x) = iter.next() {
            dbg!(x);
            let y = dbg!(x.to_bf());

            while bf_index + y.len() > self.interp.array.len() {
                self.interp.array.push(0);
            }

            if dbg!(self.interp.array.get(bf_index..(bf_index + y.len()))).unwrap() != y {
                dbg!("4");
                return false;
            }

            bf_index += y.len();
        }

        if let Some(array) = self.interp.array.get(bf_index..) {
            if array.iter().any(|x| *x != 0) {
                dbg!("5");
                return false;
            }
        }

        true
    }
}

impl Default for TestComplier {
    fn default() -> Self {
        Self {
            interp: Default::default(),
            enabled: true,
            array: Default::default(),
        }
    }
}

impl BFAsmComplier for TestComplier {
    fn get_mut(&mut self, pos: usize, len: usize) -> &mut [PrimativeType] {
        self.array.get_mut(pos, len)
    }

    fn index(&mut self) -> &mut usize {
        &mut self.array.index
    }

    fn get_input(&mut self) -> u8 {
        self.array.get_input()
    }

    fn push_output(&mut self, val: u8) {
        self.array.push_output(val);
    }

    fn write(&mut self, code: &str) {
        if self.enabled {
            self.interp
                .instructs
                .append(&mut BFInstructions::from_str(code));
        }
    }

    fn label(&mut self, name: &str) {
        if self.enabled {
            self.interp
                .instructs
                .push(BFInstructions::Label(String::from(name)));
        }

        let ret = self.interp.exec_until_label().unwrap();

        assert_eq!(ret, name);

        //TODO add cmp of the arrays
        assert!(self.compare_states())
    }

    fn enabled(&mut self) -> &mut bool {
        &mut self.enabled
    }
}

// might split into a traverse over [PrimativeValue] method
fn move_ptr_to(compiler: &mut impl BFAsmComplier, target: usize) {
    // this function always assumes the bf pointer is on the last cell of the value
    // this is only important on multi cell values like lists and in the future signed ints
    while *compiler.index() != target {
        if *compiler.index() < target {
            // moving left to right
            let index = *compiler.index();
            let x = compiler.get_mut(index, 1)[0].traverse().0;
            compiler.write(x);
            *compiler.index() += 1;
        } else if *compiler.index() > target {
            //moving right to left
            let index = *compiler.index();
            let x = compiler.get_mut(index - 1, 1)[0].traverse().1;
            compiler.write(x);
            *compiler.index() -= 1;
        } else {
            panic!("???");
        }
    }
}

fn int_add(compiler: &mut impl BFAsmComplier, target: usize) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    let mut slice = compiler.get_mut_chunk(target);

    let [PrimativeType::Int(x), PrimativeType::Int(y)] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x += *y;

    slice[1] = PrimativeType::Uninit;

    compiler.write(">[-<+>]<");
    compiler.label("int_add");

    Ok(())
}

fn int_set(compiler: &mut impl BFAsmComplier, target: usize, val: u8) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    let mut slice = compiler.get_mut_chunk(target);

    let [x @ PrimativeType::Uninit] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x = PrimativeType::Int(val);

    (0..val).for_each(|_| compiler.write("+"));
    compiler.label("int_set");

    Ok(())
}

fn int_const_add(
    compiler: &mut impl BFAsmComplier,
    target: usize,
    val: u8,
) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    let mut slice = compiler.get_mut_chunk(target);

    let [PrimativeType::Int(x)] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x += val;

    (0..val).for_each(|_| compiler.write("+"));
    compiler.label("int_const_add");

    Ok(())
}

fn int_copy(compiler: &mut impl BFAsmComplier, target: usize) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    let mut slice = compiler.get_mut_chunk(target);

    let [
        PrimativeType::Int(_),
        PrimativeType::Uninit,
        PrimativeType::Uninit,
    ] = &mut slice
    else {
        return Err(CompilerError::TypeMismatch);
    };

    slice[1] = slice[0].clone();

    // todo this clone should be unnessicary
    slice[2] = slice[0].clone();

    slice[0] = PrimativeType::Uninit;

    compiler.write("[->+>+<<]");
    compiler.label("int_copy");

    Ok(())
}

fn int_remove(compiler: &mut impl BFAsmComplier, target: usize) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    let mut slice = compiler.get_mut_chunk(target);

    let [x @ PrimativeType::Int(_)] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x = PrimativeType::Uninit;

    compiler.write("[-]");
    compiler.label("int_remove");

    Ok(())
}

fn bool_set(
    compiler: &mut impl BFAsmComplier,
    target: usize,
    val: bool,
) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    let mut slice = compiler.get_mut_chunk(target);

    let [x @ PrimativeType::Uninit] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x = PrimativeType::Boolean(val);

    if val {
        compiler.write("+");
    }
    compiler.label("bool_set");

    Ok(())
}

fn bool_copy(compiler: &mut impl BFAsmComplier, target: usize) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    let mut slice = compiler.get_mut_chunk(target);

    let [
        PrimativeType::Boolean(_),
        PrimativeType::Uninit,
        PrimativeType::Uninit,
    ] = &mut slice
    else {
        return Err(CompilerError::TypeMismatch);
    };

    slice[1] = slice[0].clone();

    // todo this clone should be unnessicary
    slice[2] = slice[0].clone();

    slice[0] = PrimativeType::Uninit;

    compiler.write("[->+>+<<]");
    compiler.label("bool_copy");

    Ok(())
}

fn bool_remove(compiler: &mut impl BFAsmComplier, target: usize) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    let mut slice = compiler.get_mut_chunk(target);

    let [x @ PrimativeType::Boolean(_)] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x = PrimativeType::Uninit;

    compiler.write("[-]");
    compiler.label("bool_remove");

    Ok(())
}

// BoolNot(usize),
// While(usize, Vec<Instruction>),
fn input(compiler: &mut impl BFAsmComplier, target: usize) -> Result<(), CompilerError> {
    let val = compiler.get_input();

    move_ptr_to(compiler, target);

    let mut slice = compiler.get_mut_chunk(target);

    let [x @ PrimativeType::Uninit] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x = PrimativeType::Int(val);

    compiler.write(",");

    compiler.label("input");

    Ok(())
}

fn output(compiler: &mut impl BFAsmComplier, target: usize) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    let mut slice = compiler.get_mut_chunk(target);

    let [PrimativeType::Int(x)] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    let x = *x;

    compiler.push_output(x);

    compiler.write(".");
    compiler.label("output");

    Ok(())
}

fn main() {
    use BFAsmInstruction as BFASM;
    use InstructionVariant as Var;

    // insert testcase here
    let mut test = TestComplier::default();

    test.exec_instructs(&[
        BFASM::new(0, Var::IntSet(2)),
        BFASM::new(1, Var::IntSet(2)),
        BFASM::new(0, Var::IntAdd),
    ]);

    dbg!(test);
}

#[cfg(test)]
mod test {
    use super::*;

    use super::BFAsmInstruction as BFASM;
    use super::InstructionVariant as Var;

    #[test]
    fn int_add() {
        let mut test = TestComplier::default();

        test.exec_instructs(&[
            BFASM::new(0, Var::IntSet(2)),
            BFASM::new(1, Var::IntSet(2)),
            BFASM::new(0, Var::IntAdd),
        ]);

        dbg!(test);
    }

    #[test]
    fn upper_to_lower() {
        let mut test = TestComplier::default();

        test.push_input('A' as u32 as u8);

        test.exec_instructs(&[
            BFASM::new(0, Var::Input),
            BFASM::new(0, Var::IntConstAdd(32)),
            BFASM::new(0, Var::Output),
        ]);

        dbg!(test);
    }
}
