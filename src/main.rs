use std::collections::{HashMap, HashSet};

use crate::bfasm::{DebugComplier, Empty};

type EmptyType = bfasm::PrimativeType<bfasm::Empty, bfasm::EmptyVec>;

mod bf;
mod bfasm;

#[derive(Debug, Clone)]
enum Token {
    OpenBrace,
    CloseBrace,
    OpenParens,
    CloseParens,

    SemiColon,
    Equals,
    Path,
    Star,
    Comma,

    Plus,
    Minus,
    Not,
    Dot,

    LineComment(String),

    Mod,
    Let,
    Mut,
    Fn,
    While,
    If,
    Use,

    Name(String),
    Literal(Literal),
}

#[derive(Debug, Clone)]
enum Literal {
    Int(u8),
    Bool(bool),
}

fn tokenize_str(name: &str) -> Option<Token> {
    if name.chars().next().unwrap().is_ascii_digit() {
        return Some(Token::Literal(Literal::Int(name.parse().unwrap())));
    }

    match name {
        "let" => Some(Token::Let),
        "mut" => Some(Token::Mut),
        "fn" => Some(Token::Fn),
        "mod" => Some(Token::Mod),
        "while" => Some(Token::While),
        "true" => Some(Token::Literal(Literal::Bool(true))),
        "false" => Some(Token::Literal(Literal::Bool(false))),
        "if" => Some(Token::If),
        "use" => Some(Token::Use),

        _ => None,
    }
}

fn tokenize(program_txt: &str) -> Vec<Token> {
    let mut program = program_txt.chars();

    let mut tokens = Vec::new();

    let mut name: Option<String> = None;

    while let Some(car) = program.next() {
        if let 'a'..='z' | 'A'..='Z' | '0'..='9' | '_' = car {
            // add to the current ident
            if let Some(x) = &mut name {
                x.push(car)
            } else {
                name = Some(String::from(car))
            }

            continue;
        } else {
            // clear current name

            if let Some(x) = name {
                if let Some(token) = tokenize_str(&x) {
                    tokens.push(token);
                } else {
                    tokens.push(Token::Name(x));
                }
            }

            name = None;
        }

        match car {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => {
                unreachable!()
            }

            ' ' | '\n' => {}

            '{' => tokens.push(Token::OpenBrace),
            '}' => tokens.push(Token::CloseBrace),
            '(' => tokens.push(Token::OpenParens),
            ')' => tokens.push(Token::CloseParens),

            ';' => tokens.push(Token::SemiColon),
            '=' => tokens.push(Token::Equals),

            '/' => {
                assert_eq!('/', program.next().unwrap());

                let mut comment = String::new();

                loop {
                    let car = program.next().unwrap();

                    if car == '\n' {
                        break;
                    } else {
                        comment.push(car);
                    }
                }

                tokens.push(Token::LineComment(comment));
            }

            ':' => {
                assert_eq!(':', program.next().unwrap());

                tokens.push(Token::Path);
            }

            '*' => tokens.push(Token::Star),
            ',' => tokens.push(Token::Comma),

            '+' => tokens.push(Token::Plus),
            '-' => tokens.push(Token::Minus),
            '!' => tokens.push(Token::Not),
            '.' => tokens.push(Token::Dot),

            _ => {
                dbg!(car);
                todo!()
            }
        }
    }

    // clean up vars
    if let Some(x) = name {
        if let Some(token) = tokenize_str(&x) {
            tokens.push(token);
        } else {
            tokens.push(Token::Name(x));
        }
    }
    tokens
}

#[derive(Debug, Clone)]
struct AST {
    imports: Vec<String>,
    functions: Vec<Function>,
}

#[derive(Debug, Clone)]
struct Function {
    name: String,
    sig: FunctionSig,
    statements: Vec<Statement>,
    requirements: HashSet<String>,
}

#[derive(Debug, Clone)]
struct FunctionSig {
    args: Vec<EmptyType>,
    ret: EmptyType,
}

#[derive(Debug, Clone)]
enum Statement {
    Declaration {
        variable: String,
        value: Value,
    },
    Assignment {
        variable: String,
        value: Value,
    },

    While {
        statements: Vec<Statement>,
        value: Value,
    },

    If {
        statements: Vec<Statement>,
        value: Value,
    },

    // value should be uninit
    Unit {
        value: Value,
    },
}

#[derive(Debug, Clone)]
struct Value {
    // val_type: EmptyType,
    val_source: ValueSource,
}

#[derive(Debug, Clone)]
enum ValueSource {
    Constant(Literal),
    Variable { name: String },
    // FunctionCall { name: String },
    Operation(Operation),
}
#[derive(Debug, Clone)]
enum Operation {
    IntAdd(Box<Value>, Box<Value>),
    IntSub(Box<Value>, u8),
    IntEq(Box<Value>, Box<Value>),
    BoolNot(Box<Value>),
}

// TODO we need to generate at least the function definition before parseing main if it contains
// another user defined function for type labling
fn tokens_to_ast(raw_tokens: Vec<Token>) -> AST {
    let mut tokens = raw_tokens.into_iter();

    let mut functions = Vec::new();

    let mut _imports = Vec::new();

    while let Some(token) = tokens.next() {
        match token {
            Token::Fn => {
                let Some(Token::Name(name)) = tokens.next() else {
                    panic!()
                };

                let sig = parse_fn_def(&mut tokens);

                let mut requirements = HashSet::new();

                let statements = parse_statements(&mut tokens, &mut requirements);

                let func = Function {
                    name,
                    sig,
                    statements,
                    requirements,
                };

                functions.push(func);
            }
            Token::Mod => {
                let Some(Token::Name(_name)) = tokens.next() else {
                    panic!()
                };
                let Some(Token::SemiColon) = tokens.next() else {
                    panic!()
                };

                // TODO
            }

            Token::LineComment(_) => {}

            Token::Use => {
                let Some(Token::Name(_name)) = tokens.next() else {
                    panic!()
                };

                let Some(Token::Path) = tokens.next() else {
                    panic!()
                };

                let Some(Token::Star) = tokens.next() else {
                    panic!()
                };

                let Some(Token::SemiColon) = tokens.next() else {
                    panic!()
                };

                // TODO
            }

            _ => unreachable!(),
        }
    }

    AST {
        imports: _imports,
        functions,
    }
}

fn parse_fn_def(tokens: &mut std::vec::IntoIter<Token>) -> FunctionSig {
    let Some(Token::OpenParens) = tokens.next() else {
        panic!()
    };

    // TODO parse args

    let Some(Token::CloseParens) = tokens.next() else {
        panic!()
    };

    let Some(Token::OpenBrace) = tokens.next() else {
        panic!()
    };

    FunctionSig {
        args: Vec::new(),
        ret: EmptyType::Uninit,
    }
}

fn parse_statements(
    tokens: &mut std::vec::IntoIter<Token>,
    reqs: &mut HashSet<String>,
) -> Vec<Statement> {
    let mut top_level_statements = Vec::new();

    loop {
        match tokens.next().unwrap() {
            Token::Name(name) => {
                let token = tokens.next().unwrap();

                if let Token::Equals = token {
                    let (value, token) = parse_value(tokens, reqs);

                    let Token::SemiColon = token else { panic!() };

                    top_level_statements.push(Statement::Assignment {
                        variable: name,
                        value,
                    })
                } else if let Token::Dot = token {
                    let Token::OpenParens = tokens.next().unwrap() else {
                        panic!()
                    };

                    let Token::Name(fn_name) = tokens.next().unwrap() else {
                        panic!()
                    };

                    reqs.insert(fn_name.clone());

                    let mut args = parse_fn_args(tokens);

                    args.insert(
                        0,
                        Value {
                            val_source: ValueSource::Variable { name },
                        },
                    );

                    // TODO add chaining dot operator
                    let Token::SemiColon = tokens.next().unwrap() else {
                        panic!()
                    };
                } else if let Token::OpenParens = token {
                    todo!();
                    reqs.insert(name.clone());
                } else {
                    dbg!(token);
                    unreachable!();
                }
            }
            Token::Let => {
                let mut token = tokens.next().unwrap();

                let mutable = if let Token::Mut = &token {
                    token = tokens.next().unwrap();
                    true
                } else {
                    false
                };

                let Token::Name(variable) = token else {
                    panic!()
                };

                let Token::Equals = tokens.next().unwrap() else {
                    panic!()
                };

                let (value, token) = parse_value(tokens, reqs);

                let Token::SemiColon = token else { panic!() };

                // TODO make copied work
                top_level_statements.push(Statement::Declaration { variable, value })
            }

            Token::CloseBrace => {
                break;
            }

            Token::LineComment(_) => {}

            Token::While => {
                let (value, Token::OpenBrace) = parse_value(tokens, reqs) else {
                    panic!()
                };

                let statements = parse_statements(tokens, reqs);

                top_level_statements.push(Statement::While {
                    statements: parse_statements(tokens, reqs),
                    value: value,
                });
            }

            Token::If => {
                let (value, Token::OpenBrace) = parse_value(tokens, reqs) else {
                    panic!()
                };

                top_level_statements.push(Statement::If {
                    statements: parse_statements(tokens, reqs),
                    value: value,
                });
            }

            _ => unreachable!(),
        }
    }

    return top_level_statements;
}

fn parse_value(
    tokens: &mut std::vec::IntoIter<Token>,
    reqs: &mut HashSet<String>,
) -> (Value, Token) {
    // let mut expressions = Vec::new();

    let value = match tokens.next().unwrap() {
        // either a function or variable
        Token::Name(name) => Value {
            val_source: ValueSource::Variable { name },
        },
        Token::Literal(lit) => match lit {
            Literal::Int(_) => Value {
                val_source: ValueSource::Constant(lit),
            },
            Literal::Bool(_) => Value {
                val_source: ValueSource::Constant(lit),
            },
        },

        Token::Not => {
            // TODO this currently assumes that the not ends the line which might not be true if we
            // add binops

            let (value, token) = parse_value(tokens, reqs);

            return (
                Value {
                    val_source: ValueSource::Operation(Operation::BoolNot(Box::new(value))),
                },
                token,
            );
        }

        Token::OpenParens => {
            let (value, Token::CloseParens) = parse_value(tokens, reqs) else {
                panic!()
            };

            value
        }

        x => {
            dbg!(x);
            unimplemented!()
        }
    };

    let token = tokens.next().unwrap();

    // if let Token::SemiColon = token {
    //     return (value, Token::SemiColon);
    // };

    match token {
        Token::Plus => {
            let (value2, token) = parse_value(tokens, reqs);

            (
                Value {
                    // TODO fix this to not assume plus acts on ints
                    val_source: ValueSource::Operation(Operation::IntAdd(
                        Box::new(value),
                        Box::new(value2),
                    )),
                },
                token,
            )
        }

        Token::Equals => {
            let Token::Equals = tokens.next().unwrap() else {
                panic!()
            };

            let (value2, token) = parse_value(tokens, reqs);

            (
                Value {
                    val_source: ValueSource::Operation(Operation::IntEq(
                        Box::new(value),
                        Box::new(value2),
                    )),
                },
                token,
            )
        }

        Token::Minus => {
            let (
                Value {
                    val_source: ValueSource::Constant(Literal::Int(x)),
                },
                token,
            ) = parse_value(tokens, reqs)
            else {
                panic!()
            };

            (
                Value {
                    val_source: ValueSource::Operation(Operation::IntSub(Box::new(value), x)),
                },
                token,
            )
        }

        x @ (Token::OpenBrace | Token::SemiColon | Token::CloseParens | Token::Comma) => {
            return (value, x);
        }

        x => {
            dbg!(x);
            todo!()
        }
    }
}

fn parse_fn_args(tokens: &mut std::vec::IntoIter<Token>) -> Vec<Value> {
    todo!()
}

#[derive(Debug, Clone)]
struct TypedAST {
    imports: Vec<String>,
    functions: Vec<TypedFunction>,
}

#[derive(Debug, Clone)]
struct TypedFunction {
    name: String,
    sig: FunctionSig,
    statements: Vec<TypedStatement>,
    requirements: HashSet<String>,
}

#[derive(Debug, Clone)]
enum TypedStatement {
    Declaration {
        _mutable: bool,
        copied: bool,
        variable: String,
        value: TypedValue,
    },
    Assignment {
        variable: String,
        value: TypedValue,
    },

    While {
        statements: Vec<TypedStatement>,
        value: TypedValue,
    },

    If {
        statements: Vec<TypedStatement>,
        value: TypedValue,
    },

    // value should be uninit
    Unit {
        value: TypedValue,
    },
}

#[derive(Debug, Clone)]
struct TypedValue {
    val_type: EmptyType,
    val_source: TypedValueSource,
}

#[derive(Debug, Clone)]
enum TypedValueSource {
    Constant(Literal),
    Variable { name: String },
    // FunctionCall { name: String },
    Operation(TypedOperation),
}
#[derive(Debug, Clone)]
enum TypedOperation {
    IntAdd(Box<TypedValue>, Box<TypedValue>),
    IntSub(Box<TypedValue>, u8),
    IntEq(Box<TypedValue>, Box<TypedValue>),
    BoolNot(Box<TypedValue>),
}

fn type_ast(mut ast: AST) -> TypedAST {
    // let main = ast.functions.iter().find(|x| x.name == "main").unwrap();

    let mut queue = vec![String::from("main")];

    let mut x = 0;

    // TODO make this detect recursion
    while x < queue.len() {
        let fn_name = queue.get(x).unwrap();

        let prereqs = ast
            .functions
            .iter()
            .find(|x| x.name == **fn_name)
            .unwrap()
            .requirements
            .iter();

        prereqs.for_each(|func| {
            if !queue.contains(func) {
                queue.push(func.clone());
            }
        });

        x += 1;
    }

    let mut sigs = libf_func_sigs();

    let mut functions = Vec::new();

    while let Some(str) = queue.pop() {
        let index = ast
            .functions
            .iter()
            .enumerate()
            .find(|(x, y)| y.name == str)
            .unwrap()
            .0;

        let func = ast.functions.remove(index);

        functions.push(type_function(func, &mut sigs))
    }

    assert_eq!(ast.functions.len(), 0);

    TypedAST {
        imports: ast.imports,
        functions,
    }
}

fn libf_func_sigs() -> HashMap<String, FunctionSig> {
    todo!()
}

fn type_function(func: Function, sigs: &mut HashMap<String, FunctionSig>) -> TypedFunction {
    todo!()
}

// TODO this does not handle recursion or indirect recursion
fn ast_to_bfasm(ast: TypedAST) -> Vec<bfasm::Instruction> {
    // assert_eq!(&ast.imports, &[String::from("libf")]);

    // let mut functions: HashMap<&str, bfasm::Function> = HashMap::new();

    let main_fn = ast.functions.iter().find(|x| x.name == "main").unwrap();

    // TODO compiler mains dependences before compiling it and so on
    // or this dependency initalization could be done by ast_fn_to_bfasm

    statements_to_bfasm(
        main_fn.statements.clone(),
        &mut HashMap::new(),
        // &mut functions,
        0,
    )
}

fn statements_to_bfasm(
    statements: Vec<TypedStatement>,
    vars: &mut HashMap<String, (EmptyType, usize)>,
    // functions: &mut HashMap<&str, bfasm::Function>,
    mut offset: usize,
) -> Vec<bfasm::Instruction> {
    // let mut reverse_stack: Vec<Value> = Vec::new();
    // let mut reverse_stack: Vec<bfasm::PrimativeType<bfasm::Empty>> = Vec::new();

    let mut instructions = Vec::new();

    // usize is the pos not the padding
    // let mut vars: Vec<(String, EmptyType, usize)> = Vec::new();

    for statement in statements {
        match statement {
            TypedStatement::Declaration {
                copied,
                _mutable,
                variable,
                value,
            } => {
                instructions.append(&mut value_to_bfasm(&value, &vars, offset));

                assert!(vars.insert(variable, (value.val_type, offset)).is_none());

                // TODO offset should be directly related to vars
                // insert a check? generate the offset dynamically
                offset += 1;

                if copied {
                    offset += 2;
                }
            }
            TypedStatement::Assignment { variable, value } => {
                instructions.append(&mut value_to_bfasm(&value, &vars, offset));

                let (var_type, pos) = vars.get(&variable).unwrap();

                match var_type {
                    bfasm::PrimativeType::Int(_) => {
                        instructions.push(bfasm::Instruction::new(
                            *pos,
                            bfasm::InstructionVariant::IntRemove,
                        ));
                        instructions.push(bfasm::Instruction::new(
                            offset,
                            bfasm::InstructionVariant::IntMove(*pos),
                        ));
                    }
                    bfasm::PrimativeType::Bool(_) => {
                        instructions.push(bfasm::Instruction::new(
                            *pos,
                            bfasm::InstructionVariant::BoolRemove,
                        ));
                        instructions.push(bfasm::Instruction::new(
                            offset,
                            bfasm::InstructionVariant::BoolMove(*pos),
                        ));
                    }
                    bfasm::PrimativeType::Uninit => unreachable!(),
                    bfasm::PrimativeType::List(_) => todo!(),
                }
            }
            TypedStatement::While { statements, value } => {
                let mut value_instructs = value_to_bfasm(&value, &vars, offset);

                instructions.append(&mut (value_instructs.clone()));

                let mut while_instructs = Vec::new();

                while_instructs.push(bfasm::Instruction::new(
                    offset,
                    bfasm::InstructionVariant::BoolRemove,
                ));

                while_instructs.append(&mut statements_to_bfasm(statements, vars, offset));

                while_instructs.append(&mut value_instructs);

                instructions.push(bfasm::Instruction::new(
                    offset,
                    bfasm::InstructionVariant::While(while_instructs),
                ));

                instructions.push(bfasm::Instruction::new(
                    offset,
                    bfasm::InstructionVariant::BoolRemove,
                ));
            }
            TypedStatement::If { statements, value } => {
                instructions.append(&mut value_to_bfasm(&value, &vars, offset));

                let mut while_instructs = Vec::new();

                while_instructs.push(bfasm::Instruction::new(
                    offset,
                    bfasm::InstructionVariant::BoolRemove,
                ));

                while_instructs.append(&mut statements_to_bfasm(statements, vars, offset));

                while_instructs.push(bfasm::Instruction::new(
                    offset,
                    bfasm::InstructionVariant::BoolSet(false),
                ));

                instructions.push(bfasm::Instruction::new(
                    offset,
                    bfasm::InstructionVariant::While(while_instructs),
                ));

                instructions.push(bfasm::Instruction::new(
                    offset,
                    bfasm::InstructionVariant::BoolRemove,
                ));
            }
            TypedStatement::Unit { value } => {
                instructions.append(&mut value_to_bfasm(&value, vars, offset));
            }
        }
    }

    instructions
}

fn value_to_bfasm(
    value: &TypedValue,
    vars: &HashMap<String, (EmptyType, usize)>,
    // reverse_stack: &mut Vec<bfasm::PrimativeType<bfasm::Empty>>,
    offset: usize,
) -> Vec<bfasm::Instruction> {
    match &value.val_source {
        TypedValueSource::Constant(Literal::Int(x)) => {
            vec![bfasm::Instruction::new(
                offset,
                bfasm::InstructionVariant::IntSet(*x),
            )]
        }

        TypedValueSource::Constant(Literal::Bool(x)) => {
            vec![bfasm::Instruction::new(
                offset,
                bfasm::InstructionVariant::BoolSet(*x),
            )]
        }

        TypedValueSource::Variable { name } => {
            let x = vars.get(name).unwrap();

            let index = x.1;

            match x.0 {
                bfasm::PrimativeType::Int(_) => {
                    vec![
                        bfasm::Instruction::new(index, bfasm::InstructionVariant::IntCopy),
                        bfasm::Instruction::new(
                            index + 1,
                            bfasm::InstructionVariant::IntMove(index),
                        ),
                        bfasm::Instruction::new(
                            index + 2,
                            bfasm::InstructionVariant::IntMove(offset),
                        ),
                    ]
                }
                bfasm::PrimativeType::Bool(_) => {
                    vec![
                        bfasm::Instruction::new(index, bfasm::InstructionVariant::BoolCopy),
                        bfasm::Instruction::new(
                            index + 1,
                            bfasm::InstructionVariant::BoolMove(index),
                        ),
                        bfasm::Instruction::new(
                            index + 2,
                            bfasm::InstructionVariant::BoolMove(offset),
                        ),
                    ]
                }
                bfasm::PrimativeType::Uninit => unreachable!(),
                bfasm::PrimativeType::List(_) => todo!(),
            }
        }
        // Value::FunctionCall { name } => todo!(),
        TypedValueSource::Operation(op) => {
            let mut instructs = Vec::new();
            match op {
                TypedOperation::IntAdd(value1, value2) => {
                    instructs.append(&mut value_to_bfasm(&value1, vars, offset));
                    instructs.append(&mut value_to_bfasm(&value2, vars, offset + 1));

                    instructs.push(bfasm::Instruction::new(
                        offset,
                        bfasm::InstructionVariant::IntAdd,
                    ));
                }

                TypedOperation::IntSub(value, num) => {
                    instructs.append(&mut value_to_bfasm(&value, vars, offset));
                    (0..*num).for_each(|_| {
                        instructs.push(bfasm::Instruction::new(
                            offset,
                            bfasm::InstructionVariant::UnsafeSub,
                        ))
                    });
                }

                TypedOperation::IntEq(value1, value2) => {
                    instructs.append(&mut value_to_bfasm(&value1, vars, offset));
                    instructs.append(&mut value_to_bfasm(&value2, vars, offset + 2));

                    instructs.push(bfasm::Instruction::new(
                        offset,
                        bfasm::InstructionVariant::IntEq,
                    ));
                }

                TypedOperation::BoolNot(value) => {
                    instructs.append(&mut value_to_bfasm(&value, vars, offset));

                    instructs.push(bfasm::Instruction::new(
                        offset,
                        bfasm::InstructionVariant::BoolNot,
                    ));
                }
            }

            instructs
        }
    }
}

fn main() {
    // bfasm::main();
    // todo!();

    let program = std::fs::read_to_string("./program/src/main.rs").unwrap();

    let tokens = tokenize(&program);

    dbg!(&tokens);

    let ast = tokens_to_ast(tokens);

    dbg!(&ast);

    // let instructs = ast_to_bfasm(ast);

    // dbg!(&instructs);

    // let mut compiler = DebugComplier::default();
    // compiler.exec_instructs(&instructs);
    // dbg!(compiler);
}

#[cfg(test)]
mod test {}
