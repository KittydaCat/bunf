use std::collections::HashMap;

use crate::bfasm::DebugComplier;

type EmptyType = bfasm::PrimativeType<bfasm::Empty>;

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

    Let,
    Mut,
    Fn,
    Mod,

    Name(String),
    Literal(Literal),
}

#[derive(Debug, Clone)]
enum Literal {
    Int(u8),
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
    functions: Vec<FunctionDec>,
}

#[derive(Debug, Clone)]
struct FunctionDec {
    name: String,
    // requirements: Vec<String>,
    statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
enum Statement {
    Declaration {
        mutable: bool,
        copied: bool,
        variable: String,
        value: Value,
    },
    Assignment {
        variable: String,
        value: Value,
    },
    // Unit {
    //     value: Value,
    // },
}

#[derive(Debug, Clone)]
enum Value {
    Constant(Literal),
    Variable { name: String },
    Function { name: String },
}

fn tokens_to_ast(raw_tokens: Vec<Token>) -> AST {
    let mut tokens = raw_tokens.into_iter();

    let mut functions = Vec::new();

    let mut imports = Vec::new();

    while let Some(token) = tokens.next() {
        match token {
            Token::Fn => {
                let name = parse_fn_args(&mut tokens);

                let statements = parse_statements(&mut tokens);

                let func = FunctionDec { name, statements };

                functions.push(func);
            }
            Token::Mod => {
                let Some(Token::Name(name)) = tokens.next() else {
                    panic!()
                };
                let Some(Token::SemiColon) = tokens.next() else {
                    panic!()
                };

                imports.push(name);
            }

            _ => unreachable!(),
        }
    }

    AST { imports, functions }
}

fn parse_fn_args(tokens: &mut std::vec::IntoIter<Token>) -> String {
    let Some(Token::Name(name)) = tokens.next() else {
        panic!()
    };

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

    return name;
}

fn parse_statements(tokens: &mut std::vec::IntoIter<Token>) -> Vec<Statement> {
    let mut statements = Vec::new();

    loop {
        match tokens.next().unwrap() {
            Token::Name(variable) => {
                let Token::Equals = tokens.next().unwrap() else {
                    panic!()
                };

                let value = parse_value(tokens);

                statements.push(Statement::Assignment { variable, value })
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

                let value = parse_value(tokens);

                // TODO make copied work

                statements.push(Statement::Declaration {
                    copied: true,
                    mutable,
                    variable,
                    value,
                })
            }

            Token::CloseBrace => {
                break;
            }

            _ => unreachable!(),
        }
    }

    return statements;
}

fn parse_value(token: &mut std::vec::IntoIter<Token>) -> Value {
    // let mut expressions = Vec::new();

    let value = match token.next().unwrap() {
        // either a function or variable
        Token::Name(name) => Value::Variable { name },
        Token::Literal(x) => Value::Constant(x),

        x => {
            dbg!(x);
            unimplemented!()
        }
    };

    let Some(Token::SemiColon) = token.next() else {
        panic!()
    };

    value
}

// TODO this does not handle recursion or indirect recursion
fn ast_to_bfasm(ast: AST) -> Vec<bfasm::Instruction> {
    assert_eq!(&ast.imports, &[String::from("libf")]);

    let mut functions: HashMap<&str, bfasm::Function> = HashMap::new();

    let main_fn = ast.functions.iter().find(|x| x.name == "main").unwrap();

    // TODO compiler mains dependences before compiling it and so on
    // or this dependency initalization could be done by ast_fn_to_bfasm

    ast_fn_to_bfasm(main_fn.clone(), &mut functions).instructions
}

fn ast_fn_to_bfasm(
    function: FunctionDec,
    functions: &mut HashMap<&str, bfasm::Function>,
) -> bfasm::Function {
    // let mut reverse_stack: Vec<Value> = Vec::new();
    // let mut reverse_stack: Vec<bfasm::PrimativeType<bfasm::Empty>> = Vec::new();

    let mut instructions = Vec::new();

    let mut vars: Vec<(String, EmptyType, usize)> = Vec::new();

    let mut offset = 0;

    let FunctionDec { name, statements } = function;

    for statement in statements {
        match statement {
            Statement::Declaration {
                copied,
                mutable,
                variable,
                value,
            } => {
                instructions.append(&mut ast_value_to_bfasm(&value, &vars, offset));

                vars.push((variable, EmptyType::Int(bfasm::Empty), offset));

                offset += 1;

                if copied {
                    offset += 2;
                }
            }
            Statement::Assignment { variable, value } => todo!(),
        }
    }

    bfasm::Function {
        argument_types: Vec::new(),
        return_types: Vec::new(),
        instructions,
    }
}

fn ast_value_to_bfasm(
    value: &Value,
    vars: &[(String, EmptyType, usize)],
    // reverse_stack: &mut Vec<bfasm::PrimativeType<bfasm::Empty>>,
    offset: usize,
) -> Vec<bfasm::Instruction> {
    match value {
        Value::Constant(Literal::Int(x)) => {
            vec![bfasm::Instruction::new(
                offset,
                bfasm::InstructionVariant::IntSet(*x),
            )]
        }
        Value::Variable { name } => {
            let x = vars.iter().find(|x| &x.0 == name).unwrap();

            let index = x.2;

            match x.1 {
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
            }
        }
        Value::Function { name } => todo!(),
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

    let instructs = ast_to_bfasm(ast);

    dbg!(&instructs);

    let mut compiler = DebugComplier::default();

    compiler.exec_instructs(&instructs);
}

#[cfg(test)]
mod test {}
