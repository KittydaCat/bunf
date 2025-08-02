mod bf;

use crate::bf::BFInstructions;

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
    fn exec(
        &self,
        array: &mut BFAsmArray,
        op_writer: &mut Option<&mut impl BFWrite>,
    ) -> Result<(), CompilerError> {
        let target = self.target;
        match &self.variant {
            InstructionVariant::IntSet(val) => int_set(array, op_writer, target, *val),
            InstructionVariant::IntAdd => int_add(array, op_writer, target),
            InstructionVariant::IntConstAdd(val) => int_const_add(array, op_writer, target, *val),
            InstructionVariant::IntCopy => int_copy(array, op_writer, target),
            InstructionVariant::IntRemove => int_remove(array, op_writer, target),
            InstructionVariant::BoolSet(val) => bool_set(array, op_writer, target, *val),
            InstructionVariant::BoolNot => todo!(),
            InstructionVariant::BoolCopy => bool_copy(array, op_writer, target),
            InstructionVariant::BoolRemove => bool_remove(array, op_writer, target),
            InstructionVariant::While(_instructions) => todo!(),
            InstructionVariant::Input => input(array, op_writer, target),
            InstructionVariant::Output => output(array, op_writer, target),
        }
    }
}

#[derive(Clone, Debug)]
enum CompilerError {
    TypeMismatch,
}

// trait BFAsmComplier {
//     // array methods
//     fn get_mut(&mut self, pos: usize, len: usize) -> &mut [PrimativeType];
//
//     fn get_mut_chunk<const N: usize>(&mut self, pos: usize) -> &mut [PrimativeType; N] {
//         self.get_mut(pos, N).first_chunk_mut().unwrap()
//     }
//
//     // TODO Im not sure how I want to do the IO
//     fn get_input(&mut self) -> u8;
//
//     fn push_output(&mut self, val: u8);
//
//     // writer methods
//     fn write(&mut self, code: &str);
//
//     fn label(&mut self, name: &str);
//
//     fn enabled(&mut self) -> &mut bool;
//
//     // fn exec(&mut self, x: &BFAsmInstruction) -> Result<(), CompilerError>;
// }

trait BFWrite {
    // lable only be used once per not control-flow instruction
    fn lable(&mut self, code: &str, fn_name: &str);

    fn write(&mut self, code: &str);

    fn enabled(&mut self) -> &mut bool;
}

#[derive(Clone, Debug)]
struct BFInterpWriter {
    interp: bf::BFInterp,
    enabled: bool,
}

impl BFInterpWriter {
    fn push_input(&mut self, val: u8) {
        self.interp.input.push(val);
    }
}

impl Default for BFInterpWriter {
    fn default() -> Self {
        BFInterpWriter {
            interp: Default::default(),
            enabled: true,
        }
    }
}

impl BFWrite for BFInterpWriter {
    fn lable(&mut self, code: &str, fn_name: &str) {
        if self.enabled {
            self.interp
                .instructs
                .append(&mut BFInstructions::from_str(code));

            self.interp
                .instructs
                .push(BFInstructions::Label(String::from(fn_name)));
        }

        let lable = dbg!(self.interp.exec_until_label());

        assert_eq!(lable.unwrap(), fn_name);
        // TODO array and ptr cmps
    }

    fn write(&mut self, code: &str) {
        if !self.enabled {
            return;
        }

        self.interp
            .instructs
            .append(&mut BFInstructions::from_str(code));
    }

    fn enabled(&mut self) -> &mut bool {
        &mut self.enabled
    }
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

    fn output(&mut self, val: u8) {
        self.output.push(val);
    }
}

#[derive(Clone, Debug, Default)]
struct TestComplier {
    writer: BFInterpWriter,
    array: BFAsmArray,
}

impl TestComplier {
    fn exec_instruct(&mut self, instruct: &BFAsmInstruction) {
        instruct
            .exec(&mut self.array, &mut Some(&mut self.writer))
            .unwrap();
    }

    fn exec_instructs(&mut self, instructs: &[BFAsmInstruction]) {
        for x in instructs {
            self.exec_instruct(x);
        }
    }

    fn push_input(&mut self, val: u8) {
        self.array.push_input(val);
        self.writer.push_input(val);
    }
}

// might split into a traverse over [PrimativeValue] method
fn move_ptr_to(array: &mut BFAsmArray, op_writer: &mut Option<&mut impl BFWrite>, target: usize) {
    // this function always assumes the bf pointer is on the last cell of the value
    // this is only important on multi cell values like lists and in the future signed ints
    let Some(writer) = op_writer.as_mut() else {
        array.index = target;
        return;
    };

    while array.index != target {
        if array.index < target {
            // moving left to right
            let x = array.get_mut(array.index, 1)[0].traverse().0;
            writer.write(x);
            array.index += 1;
        } else if array.index > target {
            //moving right to left
            let x = array.get_mut(array.index - 1, 1)[0].traverse().1;
            writer.write(x);
            array.index -= 1;
        } else {
            panic!("???");
        }
    }
}

fn int_add(
    array: &mut BFAsmArray,
    op_writer: &mut Option<&mut impl BFWrite>,
    target: usize,
) -> Result<(), CompilerError> {
    move_ptr_to(array, op_writer, target);

    let mut slice = array.get_mut_chunk(target);

    let [PrimativeType::Int(x), PrimativeType::Int(y)] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x += *y;

    slice[1] = PrimativeType::Uninit;

    if let Some(writer) = op_writer.as_mut() {
        writer.lable(">[-<+>]<", "int_add");
    }

    Ok(())
}

fn int_set(
    array: &mut BFAsmArray,
    op_writer: &mut Option<&mut impl BFWrite>,
    target: usize,
    val: u8,
) -> Result<(), CompilerError> {
    move_ptr_to(array, op_writer, target);

    let mut slice = array.get_mut_chunk(target);

    let [x @ PrimativeType::Uninit] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x = PrimativeType::Int(val);

    if let Some(writer) = op_writer.as_mut() {
        (0..val).for_each(|_| writer.write("+"));
        writer.lable("", "int_set");
    }

    Ok(())
}

fn int_const_add(
    array: &mut BFAsmArray,
    op_writer: &mut Option<&mut impl BFWrite>,
    target: usize,
    val: u8,
) -> Result<(), CompilerError> {
    move_ptr_to(array, op_writer, target);

    let mut slice = array.get_mut_chunk(target);

    let [PrimativeType::Int(x)] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x += val;

    if let Some(writer) = op_writer.as_mut() {
        (0..val).for_each(|_| writer.write("+"));
        writer.lable("", "int_const_add");
    }

    Ok(())
}

fn int_copy(
    array: &mut BFAsmArray,
    op_writer: &mut Option<&mut impl BFWrite>,
    target: usize,
) -> Result<(), CompilerError> {
    move_ptr_to(array, op_writer, target);

    let mut slice = array.get_mut_chunk(target);

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

    if let Some(writer) = op_writer.as_mut() {
        writer.lable("[->+>+<<]", "int_copy");
    }

    Ok(())
}

fn int_remove(
    array: &mut BFAsmArray,
    op_writer: &mut Option<&mut impl BFWrite>,
    target: usize,
) -> Result<(), CompilerError> {
    move_ptr_to(array, op_writer, target);

    let mut slice = array.get_mut_chunk(target);

    let [x @ PrimativeType::Int(_)] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x = PrimativeType::Uninit;

    if let Some(writer) = op_writer.as_mut() {
        writer.lable("[-]", "int_remove");
    }

    Ok(())
}

fn bool_set(
    array: &mut BFAsmArray,
    op_writer: &mut Option<&mut impl BFWrite>,
    target: usize,
    val: bool,
) -> Result<(), CompilerError> {
    move_ptr_to(array, op_writer, target);

    let mut slice = array.get_mut_chunk(target);

    let [x @ PrimativeType::Uninit] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x = PrimativeType::Boolean(val);

    if let Some(writer) = op_writer.as_mut() {
        writer.lable(if val { "+" } else { "" }, "bool_set");
    }

    Ok(())
}

fn bool_copy(
    array: &mut BFAsmArray,
    op_writer: &mut Option<&mut impl BFWrite>,
    target: usize,
) -> Result<(), CompilerError> {
    move_ptr_to(array, op_writer, target);

    let mut slice = array.get_mut_chunk(target);

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

    if let Some(writer) = op_writer.as_mut() {
        writer.lable("[->+>+<<]", "bool_copy");
    }

    Ok(())
}

fn bool_remove(
    array: &mut BFAsmArray,
    op_writer: &mut Option<&mut impl BFWrite>,
    target: usize,
) -> Result<(), CompilerError> {
    move_ptr_to(array, op_writer, target);

    let mut slice = array.get_mut_chunk(target);

    let [x @ PrimativeType::Boolean(_)] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x = PrimativeType::Uninit;

    if let Some(writer) = op_writer.as_mut() {
        writer.lable("[-]", "bool_remove");
    }

    Ok(())
}

// BoolNot(usize),
// While(usize, Vec<Instruction>),
fn input(
    array: &mut BFAsmArray,
    op_writer: &mut Option<&mut impl BFWrite>,
    target: usize,
) -> Result<(), CompilerError> {
    let val = array.get_input();

    move_ptr_to(array, op_writer, target);

    let mut slice = array.get_mut_chunk(target);

    let [x @ PrimativeType::Uninit] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x = PrimativeType::Int(val);

    if let Some(writer) = op_writer.as_mut() {
        writer.lable(",", "input");
    }

    Ok(())
}

fn output(
    array: &mut BFAsmArray,
    op_writer: &mut Option<&mut impl BFWrite>,
    target: usize,
) -> Result<(), CompilerError> {
    move_ptr_to(array, op_writer, target);

    let mut slice = array.get_mut_chunk(target);

    let [PrimativeType::Int(x)] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    let x = *x;

    array.output(x);

    if let Some(writer) = op_writer.as_mut() {
        writer.lable(".", "output");
    }

    Ok(())
}

fn main() {
    use BFAsmInstruction as BFASM;
    use InstructionVariant as Var;

    // insert testcase here
    let mut test = TestComplier::default();

    test.push_input('A' as u32 as u8);

    test.exec_instructs(&[
        BFASM::new(0, Var::Input),
        BFASM::new(0, Var::IntConstAdd(32)),
        BFASM::new(0, Var::Output),
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
