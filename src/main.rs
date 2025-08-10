mod bf;

use crate::bf::BFInstruction;

// add subtract assign
trait CompilerData:
    std::fmt::Debug + Copy + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + From<u8>
{
    // fn to_u8(&self) -> u8;

    // should return true if the cell would trigger a [ loop
    fn to_bool(&self) -> bool;
}

#[derive(Debug, Copy, Clone)]
struct Empty;

impl std::ops::Add for Empty {
    type Output = Empty;

    fn add(self, _: Self) -> Self::Output {
        Empty
    }
}

impl std::ops::Sub for Empty {
    type Output = Empty;

    fn sub(self, _: Self) -> Self::Output {
        Empty
    }
}

impl From<u8> for Empty {
    fn from(_: u8) -> Self {
        Empty
    }
}

impl CompilerData for Empty {
    fn to_bool(&self) -> bool {
        false
    }
}
impl CompilerData for u8 {
    fn to_bool(&self) -> bool {
        assert_eq!(0 < *self, 0 != *self);

        0 != *self
    }
}

#[derive(Debug, Clone)]
enum PrimativeType<T: CompilerData> {
    Int(T),
    Boolean(T),
    // List(Vec<PrimativeType>),
    Uninit,
}

impl<T: CompilerData> PrimativeType<T> {
    fn traverse(&self) -> (&'static str, &'static str) {
        match self {
            PrimativeType::Int(_) | PrimativeType::Boolean(_) | PrimativeType::Uninit => (">", "<"),
        }
    }

    fn traverse_array(array: &[Self]) -> (String, String) {
        dbg!(array);
        let ltr = array.iter().map(|x| x.traverse().0).collect();

        let rtl = array.iter().rev().map(|x| x.traverse().1).collect();

        dbg!((ltr, rtl))
    }

    fn bf_len(&self) -> usize {
        match self {
            PrimativeType::Int(_) => 1,
            PrimativeType::Boolean(_) => 1,
            PrimativeType::Uninit => 1,
        }
    }

    fn to_empty(&self) -> PrimativeType<Empty> {
        match self {
            PrimativeType::Int(_) => PrimativeType::Int(Empty),
            PrimativeType::Boolean(_) => PrimativeType::Boolean(Empty),
            PrimativeType::Uninit => PrimativeType::Uninit,
        }
    }
}

impl PrimativeType<u8> {
    // this is a butt done of heap allocs might be slow as fuck
    fn to_bf(&self) -> Vec<u8> {
        match self {
            PrimativeType::Int(x) => vec![*x],
            PrimativeType::Boolean(x) => vec![*x as u8],
            PrimativeType::Uninit => vec![0],
        }
    }
}

// basic Instructions
// the initial usize is the the stack index of the value you want to do the op on
// the args are usually passed positionally i.e. Add will check that there are 2 ints at the usize
// it is given the MoveVal and Control flow ops are the notable exceptions,
#[derive(Clone, Debug)]
pub struct Instruction {
    target: usize,
    variant: InstructionVariant,
}

impl Instruction {
    pub fn new(target: usize, variant: InstructionVariant) -> Self {
        Instruction { target, variant }
    }
}

#[derive(Clone, Debug)]
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
    IntMove(usize),
    // IntSubCons(usize),
    // Cmp(usize), // how do we want this to work?
    // Match(usize, Vec<(u8, Vec<Instructions>)>),
    // Eq(usize),

    // Boolean Ops
    BoolSet(bool),
    BoolNot,
    BoolCopy,
    BoolRemove,
    BoolMove(usize),
    // If(usize, Vec<Instructions>),
    While(Vec<Instruction>),

    // List ops
    // ListInit(usize),
    // ListIndex(usize),
    // ListLen(usize),
    // ListPop(usize),
    // ListPush(usize),

    // IO
    Input,
    Output,

    // Unsafe Ops
    UnsafeSub,
}

impl Instruction {
    fn raw_exec<C: Compiler<D>, D: CompilerData>(
        &self,
        compiler: &mut C,
    ) -> Result<(), CompilerError> {
        let target = self.target;
        match &self.variant {
            InstructionVariant::IntSet(val) => int_set(compiler, target, *val),
            InstructionVariant::IntAdd => int_add(compiler, target),
            InstructionVariant::IntConstAdd(val) => int_const_add(compiler, target, *val),
            InstructionVariant::IntCopy => int_copy(compiler, target),
            InstructionVariant::IntRemove => int_remove(compiler, target),
            InstructionVariant::IntMove(index) => int_move(compiler, target, *index),

            InstructionVariant::BoolSet(val) => bool_set(compiler, target, *val),
            InstructionVariant::BoolNot => todo!(),
            InstructionVariant::BoolCopy => bool_copy(compiler, target),
            InstructionVariant::BoolRemove => bool_remove(compiler, target),
            InstructionVariant::BoolMove(_) => todo!(),

            InstructionVariant::While(instructions) => bool_while(compiler, target, instructions),
            InstructionVariant::Input => input(compiler, target),
            InstructionVariant::Output => output(compiler, target),
            InstructionVariant::UnsafeSub => unsafe_sub(compiler, target),
        }
    }

    fn name(&self) -> &str {
        match &self.variant {
            InstructionVariant::IntSet(_) => "int_set",
            InstructionVariant::IntAdd => "int_add",
            InstructionVariant::IntConstAdd(_) => "int_const_add",
            InstructionVariant::IntCopy => "int_copy",
            InstructionVariant::IntRemove => "int_remove",
            InstructionVariant::IntMove(_) => "int_move",

            InstructionVariant::BoolSet(_) => "bool_set",
            InstructionVariant::BoolNot => todo!(),
            InstructionVariant::BoolCopy => "bool_copy",
            InstructionVariant::BoolRemove => "bool_remove",
            InstructionVariant::BoolMove(_) => "bool_move",

            InstructionVariant::While(_) => "bool_while",
            InstructionVariant::Input => "input",
            InstructionVariant::Output => "output",
            InstructionVariant::UnsafeSub => "unsafe_sub",
        }
    }
}

#[derive(Clone, Debug)]
enum CompilerError {
    TypeMismatch,
}

trait Compiler<Data: CompilerData> {
    // array methods
    fn get_array_mut(&mut self) -> &mut Vec<PrimativeType<Data>>;

    fn get_mut(&mut self, pos: usize, len: usize) -> &mut [PrimativeType<Data>] {
        let array = self.get_array_mut();

        while pos + len > array.len() {
            array.push(PrimativeType::Uninit);
        }

        array.get_mut(pos..(pos + len)).unwrap()
    }

    fn get_mut_chunk<const N: usize>(&mut self, pos: usize) -> &mut [PrimativeType<Data>; N] {
        self.get_mut(pos, N).first_chunk_mut().unwrap()
    }

    fn trim_back(&mut self) {
        let array = self.get_array_mut();
        while let Some(PrimativeType::Uninit) = array.last() {
            array.pop();
        }
    }

    fn index(&mut self) -> &mut usize;

    // TODO Im not sure how I want to do the IO
    fn get_input(&mut self) -> Data;

    fn push_output(&mut self, data: Data);

    // writer methods
    fn write_instruct(&mut self, code: BFInstruction);

    fn write_bf(&mut self, code: Vec<BFInstruction>) {
        // dbg!(&code);

        if !*self.write_enabled() {
            dbg!("the thing");
            return;
        }

        code.into_iter().for_each(|x| self.write_instruct(x))
    }

    fn write(&mut self, code: &str) {
        if *self.write_enabled() {
            self.write_bf(BFInstruction::from_str(code));
        }
    }

    fn label(&mut self, name: &str) {
        // dbg!("LABEL", name);

        if *self.write_enabled() {
            self.write_instruct(BFInstruction::Label(String::from(name)))
        }
    }

    fn write_enabled(&mut self) -> &mut bool;

    fn exec(&mut self, x: &Instruction) -> Result<(), CompilerError>;

    fn exec_instructs(&mut self, instructs: &[Instruction]) {
        instructs.iter().for_each(|x| self.exec(x).unwrap())
    }
}

#[derive(Clone, Debug)]
pub struct TestComplier {
    // compiler stats
    array: Vec<PrimativeType<u8>>,
    index: usize,
    input: Vec<u8>,
    output: Vec<u8>,
    write_enabled: bool,

    interp: bf::BFInterp,
}

impl TestComplier {
    pub fn push_input(&mut self, val: u8) {
        self.input.push(val);
        self.interp.input.push(val);
    }

    fn compare_states(&mut self) -> bool {
        dbg!("comparing states");
        dbg!(&self);

        if self.input != self.interp.input {
            dbg!("1");
            return false;
        }

        if self.output != self.interp.output {
            dbg!("2");
            return false;
        }

        let array_pt_pos: usize = self.get_mut(0, self.index).iter().map(|x| x.bf_len()).sum();

        if array_pt_pos != self.interp.index {
            dbg!("3");
            return false;
        }

        self.trim_back();

        let mut iter = self.array.iter();

        let mut bf_index = 0;

        while let Some(x) = iter.next() {
            dbg!(x, bf_index);
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
            write_enabled: true,
            array: Default::default(),
            index: Default::default(),
            input: Default::default(),
            output: Default::default(),
        }
    }
}

impl Compiler<u8> for TestComplier {
    fn get_array_mut(&mut self) -> &mut Vec<PrimativeType<u8>> {
        &mut self.array
    }

    fn index(&mut self) -> &mut usize {
        &mut self.index
    }

    // TODO check the if input and output fns are correct
    fn get_input(&mut self) -> u8 {
        self.input.pop().unwrap()
    }

    fn push_output(&mut self, val: u8) {
        self.output.push(val);
    }

    fn write_instruct(&mut self, code: BFInstruction) {
        if !self.write_enabled {
            panic!()
        }
        self.interp.instructs.push(code);
    }

    fn label(&mut self, name: &str) {
        if self.write_enabled {
            self.interp
                .instructs
                .push(BFInstruction::Label(String::from(name)));
        }
    }

    fn write_enabled(&mut self) -> &mut bool {
        &mut self.write_enabled
    }

    fn exec(&mut self, x: &Instruction) -> Result<(), CompilerError> {
        dbg!(x);
        x.raw_exec(self)?;

        let label = self.interp.exec_until_label().unwrap();

        assert_eq!(x.name(), label);

        assert!(self.compare_states());

        Ok(())
    }
}

pub struct BasicCompiler {
    array: Vec<PrimativeType<Empty>>,
    index: usize,
    instructs: Vec<BFInstruction>,
    write_enabled: bool,
}

impl BasicCompiler {
    fn new<D: CompilerData>(types: &[PrimativeType<D>], index: usize) -> Self {
        let array = types.iter().map(|x| x.to_empty()).collect();
        BasicCompiler {
            array,
            index,
            ..Default::default()
        }
    }
}

impl Default for BasicCompiler {
    fn default() -> Self {
        BasicCompiler {
            array: Default::default(),
            index: Default::default(),
            instructs: Default::default(),
            write_enabled: true,
        }
    }
}

impl Compiler<Empty> for BasicCompiler {
    fn get_array_mut(&mut self) -> &mut Vec<PrimativeType<Empty>> {
        &mut self.array
    }

    fn index(&mut self) -> &mut usize {
        &mut self.index
    }

    fn get_input(&mut self) -> Empty {
        Empty
    }

    fn push_output(&mut self, _: Empty) {}

    fn write_instruct(&mut self, code: BFInstruction) {
        if !self.write_enabled {
            panic!()
        }

        self.instructs.push(code);
    }

    fn write_enabled(&mut self) -> &mut bool {
        &mut self.write_enabled
    }

    fn exec(&mut self, x: &Instruction) -> Result<(), CompilerError> {
        x.raw_exec(self)
    }
}

// might split into a traverse over [PrimativeValue] method
fn move_ptr_to<C: Compiler<D>, D: CompilerData>(compiler: &mut C, target: usize) {
    // this function always assumes the bf pointer is on the last cell of the value
    // this is only important on multi cell values like lists and in the future signed ints

    // while *compiler.index() != target {
    //     if *compiler.index() < target {
    //         // moving left to right
    //         let index = *compiler.index();
    //         let x = compiler.get_mut(index, 1)[0].traverse().0;
    //         compiler.write(x);
    //         *compiler.index() += 1;
    //     } else if *compiler.index() > target {
    //         //moving right to left
    //         let index = *compiler.index();
    //         let x = compiler.get_mut(index - 1, 1)[0].traverse().1;
    //         compiler.write(x);
    //         *compiler.index() -= 1;
    //     } else {
    //         panic!("???");
    //     }
    // }

    let index = *compiler.index();

    let code = if index < target {
        PrimativeType::traverse_array(compiler.get_mut(index, target - index)).0
    } else if index > target {
        PrimativeType::traverse_array(compiler.get_mut(target, index - target)).1
    } else {
        return;
    };

    compiler.write(&code);

    *compiler.index() = target;
}

fn int_add<C: Compiler<D>, D: CompilerData>(
    compiler: &mut C,
    target: usize,
) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    let mut slice = compiler.get_mut_chunk(target);

    let [PrimativeType::Int(x), PrimativeType::Int(y)] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x = *x + *y;

    slice[1] = PrimativeType::Uninit;

    compiler.write(">[-<+>]<");
    compiler.label("int_add");

    Ok(())
}

fn int_set<D: CompilerData, C: Compiler<D>>(
    compiler: &mut C,
    target: usize,
    val: u8,
) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    let mut slice = compiler.get_mut_chunk(target);

    let [x @ PrimativeType::Uninit] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x = PrimativeType::Int(val.into());

    (0..val).for_each(|_| compiler.write("+"));
    compiler.label("int_set");

    Ok(())
}

fn int_const_add<D: CompilerData, C: Compiler<D>>(
    compiler: &mut C,
    target: usize,
    val: u8,
) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    let mut slice = compiler.get_mut_chunk(target);

    let [PrimativeType::Int(x)] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x = *x + val.into();

    (0..val).for_each(|_| compiler.write("+"));
    compiler.label("int_const_add");

    Ok(())
}

fn int_copy<D: CompilerData, C: Compiler<D>>(
    compiler: &mut C,
    target: usize,
) -> Result<(), CompilerError> {
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

fn int_remove<D: CompilerData, C: Compiler<D>>(
    compiler: &mut C,
    target: usize,
) -> Result<(), CompilerError> {
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

fn bool_set<D: CompilerData, C: Compiler<D>>(
    compiler: &mut C,
    target: usize,
    val: bool,
) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    let mut slice = compiler.get_mut_chunk(target);

    let [x @ PrimativeType::Uninit] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x = PrimativeType::Boolean((if val { 1 } else { 0 }).into());

    if val {
        compiler.write("+");
    }
    compiler.label("bool_set");

    Ok(())
}

fn bool_copy<D: CompilerData, C: Compiler<D>>(
    compiler: &mut C,
    target: usize,
) -> Result<(), CompilerError> {
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

fn bool_remove<D: CompilerData, C: Compiler<D>>(
    compiler: &mut C,
    target: usize,
) -> Result<(), CompilerError> {
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
fn input<D: CompilerData, C: Compiler<D>>(
    compiler: &mut C,
    target: usize,
) -> Result<(), CompilerError> {
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

fn output<D: CompilerData, C: Compiler<D>>(
    compiler: &mut C,
    target: usize,
) -> Result<(), CompilerError> {
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

// not actually limited to bools
fn bool_while<D: CompilerData, C: Compiler<D>>(
    compiler: &mut C,
    target: usize,
    instructs: &[Instruction],
) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    // first slice check
    let mut slice = compiler.get_mut_chunk(target);

    let [PrimativeType::Boolean(_) /*| PrimativeType::Int(_)*/] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    // write the instructs without executing
    // this is the only place where exec_enabled is needed
    // TODO we might move this logic to the exec method Instruction
    // let exec = *compiler.exec_enabled();
    // *compiler.exec_enabled() = false;

    compiler.write("[");

    let bf_instructs = type_check(compiler.get_array_mut(), target, instructs).unwrap();
    compiler.write_bf(bf_instructs);
    // we should move the ptr back in the type checking
    //move_ptr_to(compiler, target);
    assert_eq!(*compiler.index(), target);
    compiler.write("]");

    compiler.label("bool_while");
    // *compiler.exec_enabled() = exec;

    // exec the instructs without writing
    let write_state = *compiler.write_enabled();

    *compiler.write_enabled() = false;

    while let [PrimativeType::Boolean(val)] = //  | PrimativeType::Int(val) ] =
        compiler.get_mut_chunk(target)
    {
        if !val.to_bool() {
            break;
        }
        // instructs.iter().for_each(|x| x.exec(compiler).unwrap());

        compiler.exec_instructs(instructs);

        // this might be useless
        move_ptr_to(compiler, target);
    }

    *compiler.write_enabled() = write_state;

    Ok(())
}

fn type_check<D: CompilerData>(
    array: &Vec<PrimativeType<D>>,
    target: usize,
    instructs: &[Instruction],
) -> Result<Vec<BFInstruction>, CompilerError> {
    dbg!("type checking", instructs);
    let mut compiler = BasicCompiler::new(array, target);

    compiler.exec_instructs(instructs);
    move_ptr_to(&mut compiler, target);

    assert_eq!(target, compiler.index);

    Ok(compiler.instructs)
}

fn unsafe_sub<C: Compiler<D>, D: CompilerData>(
    compiler: &mut C,
    target: usize,
) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    // first slice check
    let mut slice = compiler.get_mut_chunk(target);

    let [PrimativeType::Int(x)] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    *x = *x - 1.into();

    compiler.write("-");
    compiler.label("unsafe_sub");

    Ok(())
}

fn int_move<C: Compiler<D>, D: CompilerData>(
    compiler: &mut C,
    target: usize,
    index: usize,
) -> Result<(), CompilerError> {
    move_ptr_to(compiler, target);

    // first slice check
    let mut slice = compiler.get_mut_chunk(target);

    let [PrimativeType::Int(x)] = &mut slice else {
        return Err(CompilerError::TypeMismatch);
    };

    let val = *x;

    let mut index_slice = compiler.get_mut_chunk(index);

    let [y @ PrimativeType::Uninit] = &mut index_slice else {
        // dbg!();
        return Err(CompilerError::TypeMismatch);
    };

    *y = PrimativeType::Int(val);

    *compiler.get_mut_chunk(target) = [PrimativeType::Uninit];

    compiler.write("[-");
    // !!! index is the target pos of the move
    // !!! target is the current index of the ptr
    //
    // naming is hard
    if index < target {
        //                  \/ we are here
        // [index] _ _ _ [target]
        let (ltr, rtl) = PrimativeType::traverse_array(compiler.get_mut(index, target - index));
        compiler.write(&rtl);
        compiler.write("+");
        compiler.write(&ltr);
    } else if index > target {
        // [target] _ _ [index]
        let (ltr, rtl) = PrimativeType::traverse_array(compiler.get_mut(target, index - target));
        compiler.write(&ltr);
        compiler.write("+");
        compiler.write(&rtl);
    } else {
        panic!();
    };

    compiler.write("]");

    compiler.label("int_move");

    Ok(())
}

pub fn test(instructs: &[Instruction], input: &str) -> String {
    let mut test = TestComplier::default();

    input.chars().for_each(|x| test.push_input(x as u32 as u8));

    test.exec_instructs(instructs);

    dbg!(&test);

    BFInstruction::to_str(&test.interp.instructs)
}

fn main() {
    use Instruction as BFASM;
    use InstructionVariant as Var;
    // insert testcase here

    dbg!(test(
        &[
        // fill out
        ],
        ""
    ));
}

#[cfg(test)]
mod test {
    use super::*;

    use super::Instruction as BFASM;
    use super::InstructionVariant as Var;
    /*
    #[test]
    fn test_name() {
        assert_eq!(
            test(
                &[
                ],
                ""
            ),
            "",
        );
    }
    */

    #[test]
    fn int_copy_and_move() {
        assert_eq!(
            "+++int_set>>>++int_set<<<[->+>+<<]int_copy>>[-<<+>>]int_move<[->>>+<<<]int_move>>>[-<+>]<int_add",
            test(
                &[
                    BFASM::new(0, Var::IntSet(3)),
                    BFASM::new(3, Var::IntSet(2)),
                    BFASM::new(0, Var::IntCopy),
                    BFASM::new(2, Var::IntMove(0)),
                    BFASM::new(1, Var::IntMove(4)),
                    BFASM::new(3, Var::IntAdd),
                ],
                ""
            )
        )
    }

    #[test]
    fn int_add() {
        assert_eq!(
            test(
                &[
                    BFASM::new(0, Var::IntSet(2)),
                    BFASM::new(1, Var::IntSet(2)),
                    BFASM::new(0, Var::IntAdd),
                ],
                ""
            ),
            "++int_set>++int_set<>[-<+>]<int_add",
        );
    }

    #[test]
    fn upper_to_lower() {
        assert_eq!(
            test(
                &[
                    BFASM::new(0, Var::Input),
                    BFASM::new(0, Var::IntConstAdd(32)),
                    BFASM::new(0, Var::Output),
                ],
                "a"
            ),
            ",input++++++++++++++++++++++++++++++++int_const_add.output",
        );
    }

    #[test]
    fn if_test() {
        assert_eq!(
            test(
                &[
                    BFASM::new(0, Var::BoolSet(true)),
                    BFASM::new(1, Var::IntSet(3)),
                    BFASM::new(
                        0,
                        Var::While(vec![
                            BFASM::new(0, Var::BoolRemove),
                            BFASM::new(0, Var::BoolSet(false)),
                            BFASM::new(1, Var::IntConstAdd(3)),
                        ]),
                    ),
                ],
                ""
            ),
            "+bool_set>+++int_set<[[-]bool_removebool_set>+++int_const_add<]bool_while"
        );
    }

    #[test]
    fn nested_while() {
        assert_eq!(
            test(
                &[
                    BFASM::new(0, Var::BoolSet(true)),  // x = true
                    BFASM::new(1, Var::IntSet(3)),      // z = 3
                    BFASM::new(2, Var::BoolSet(false)), // y = false
                    BFASM::new(
                        0,
                        Var::While(vec![
                            // while x
                            BFASM::new(1, Var::IntConstAdd(2)), // z += 2
                            BFASM::new(
                                2,
                                Var::While(
                                    // if y
                                    vec![
                                        BFASM::new(2, Var::BoolRemove),
                                        BFASM::new(2, Var::BoolSet(false)),
                                        BFASM::new(0, Var::BoolRemove), // x = false
                                        BFASM::new(0, Var::BoolSet(false)),
                                    ]
                                )
                            ),
                            BFASM::new(2, Var::BoolRemove), // y = true
                            BFASM::new(2, Var::BoolSet(true)),
                        ]),
                    ),
                ],
                ""
            ),
            "+bool_set>+++int_set>bool_set<<[>++int_const_add>[[-]bool_removebool_set<<[-]bool_removebool_set>>]bool_while[-]bool_remove+bool_set<<]bool_while"
        )
    }
}
