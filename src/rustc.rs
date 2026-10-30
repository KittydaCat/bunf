use crate::bfasm::{DebugComplier, Empty};
use std::collections::{HashMap, HashSet};

type EmptyType = crate::bfasm::PrimativeType<crate::bfasm::Empty, crate::bfasm::EmptyVec>;

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
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

            '\'' => {
                tokens.push(Token::Literal(Literal::Int(
                    program.next().unwrap() as u32 as u8
                )));
                let str = "' as usize";
                let str_iter = str.chars();

                for x in str_iter {
                    assert_eq!(x, program.next().unwrap());
                }
            }

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

#[derive(Debug, Clone, PartialEq)]
struct FunctionSig {
    args: Vec<EmptyType>,
    ret: EmptyType,
}

#[derive(Debug, Clone)]
enum Statement {
    Declaration {
        variable: String,
        value: Value,
        mutable: bool,
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
    val_source: ValueSource,
}

#[derive(Debug, Clone)]
enum ValueSource {
    Constant(Literal),
    Variable { name: String },
    FunctionCall { name: String, args: Vec<Value> },
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
    let mut tokens = raw_tokens.into_iter().peekable();

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

fn parse_fn_def(tokens: &mut std::iter::Peekable<std::vec::IntoIter<Token>>) -> FunctionSig {
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
    tokens: &mut std::iter::Peekable<std::vec::IntoIter<Token>>,
    reqs: &mut HashSet<String>,
) -> Vec<Statement> {
    // dbg!(&tokens);

    let mut top_level_statements = Vec::new();

    loop {
        match tokens.next().unwrap() {
            Token::Name(name) => {
                match tokens.next().unwrap() {
                    Token::Equals => {
                        let (value, token) = parse_value(tokens, reqs);

                        let Token::SemiColon = token else { panic!() };

                        top_level_statements.push(Statement::Assignment {
                            variable: name,
                            value,
                        })
                    }
                    Token::Dot => {
                        let Token::Name(fn_name) = tokens.next().unwrap() else {
                            panic!()
                        };

                        reqs.insert(fn_name.clone());

                        let Token::OpenParens = tokens.next().unwrap() else {
                            panic!()
                        };

                        let mut args = parse_fn_args(tokens, reqs);

                        args.insert(
                            0,
                            Value {
                                val_source: ValueSource::Variable { name: name.clone() },
                            },
                        );

                        top_level_statements.push(Statement::Unit {
                            value: Value {
                                val_source: ValueSource::FunctionCall {
                                    name: fn_name,
                                    args,
                                },
                            },
                        });

                        // TODO add chaining dot operator
                        let Token::SemiColon = tokens.next().unwrap() else {
                            panic!()
                        };
                    }
                    Token::OpenParens => {
                        reqs.insert(name.clone());

                        let args = parse_fn_args(tokens, reqs);

                        top_level_statements.push(Statement::Unit {
                            value: Value {
                                val_source: ValueSource::FunctionCall { name, args },
                            },
                        });

                        let Token::SemiColon = tokens.next().unwrap() else {
                            panic!()
                        };
                    }

                    Token::Plus => {
                        assert_eq!(Token::Equals, tokens.next().unwrap());

                        let (value, token) = parse_value(tokens, reqs);

                        let Token::SemiColon = token else { panic!() };

                        top_level_statements.push(Statement::Assignment {
                            variable: name.clone(),
                            value: Value {
                                val_source: ValueSource::Operation(Operation::IntAdd(
                                    Box::new(Value {
                                        val_source: ValueSource::Variable { name },
                                    }),
                                    Box::new(value),
                                )),
                            },
                        });
                    }

                    Token::Minus => {
                        assert_eq!(Token::Equals, tokens.next().unwrap());

                        let Token::Literal(Literal::Int(x)) = tokens.next().unwrap() else {
                            panic!()
                        };

                        let Token::SemiColon = tokens.next().unwrap() else {
                            panic!()
                        };

                        top_level_statements.push(Statement::Assignment {
                            variable: name.clone(),
                            value: Value {
                                val_source: ValueSource::Operation(Operation::IntSub(
                                    Box::new(Value {
                                        val_source: ValueSource::Variable { name },
                                    }),
                                    x,
                                )),
                            },
                        });
                    }

                    token => {
                        dbg!(token);
                        unreachable!();
                    }
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

                top_level_statements.push(Statement::Declaration {
                    variable,
                    value,
                    mutable,
                })
            }

            Token::CloseBrace => {
                return top_level_statements;
            }

            Token::LineComment(_) => {}

            Token::While => {
                let (value, Token::OpenBrace) = parse_value(tokens, reqs) else {
                    panic!()
                };

                // let statements = parse_statements(tokens, reqs);

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

            x => unreachable!("{x:?}"),
        }
    }
}

fn parse_value(
    tokens: &mut std::iter::Peekable<std::vec::IntoIter<Token>>,
    reqs: &mut HashSet<String>,
) -> (Value, Token) {
    // let mut expressions = Vec::new();

    let mut value = match tokens.next().unwrap() {
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

        // Token::CloseParens => {
        //     return (todo!(), Token::CloseParens);
        // }
        x => {
            dbg!(x, tokens);
            unimplemented!()
        }
    };

    let mut token = tokens.next().unwrap();

    // if let Token::SemiColon = token {
    //     return (value, Token::SemiColon);
    // };

    loop {
        (value, token) = match token {
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

            Token::Not => {
                assert_eq!(Token::Equals, tokens.next().unwrap());

                let (value2, token) = parse_value(tokens, reqs);

                (
                    Value {
                        val_source: ValueSource::Operation(Operation::BoolNot(Box::new(Value {
                            val_source: ValueSource::Operation(Operation::IntEq(
                                Box::new(value),
                                Box::new(value2),
                            )),
                        }))),
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

            Token::Dot => {
                let Value {
                    val_source: ValueSource::Variable { name },
                } = value
                else {
                    panic!()
                };

                let Token::Name(fn_name) = tokens.next().unwrap() else {
                    panic!()
                };

                reqs.insert(fn_name.clone());

                let Token::OpenParens = tokens.next().unwrap() else {
                    panic!()
                };

                let mut args = parse_fn_args(tokens, reqs);

                args.insert(
                    0,
                    Value {
                        val_source: ValueSource::Variable { name: name.clone() },
                    },
                );

                (
                    Value {
                        val_source: ValueSource::FunctionCall {
                            name: fn_name,
                            args,
                        },
                    },
                    tokens.next().unwrap(),
                )
            }

            Token::OpenParens => {
                let Value {
                    val_source: ValueSource::Variable { name },
                } = value
                else {
                    panic!()
                };

                let args = parse_fn_args(tokens, reqs);

                reqs.insert(name.clone());

                (
                    Value {
                        val_source: ValueSource::FunctionCall { name, args },
                    },
                    tokens.next().unwrap(),
                )
            }

            x @ (Token::OpenBrace | Token::SemiColon | Token::CloseParens | Token::Comma) => {
                return (value, x);
            }

            x => {
                dbg!(x);
                dbg!(tokens);
                todo!()
            }
        }
    }
}

fn parse_fn_args(
    tokens: &mut std::iter::Peekable<std::vec::IntoIter<Token>>,
    reqs: &mut HashSet<String>,
) -> Vec<Value> {
    let mut values = Vec::new();

    if tokens.peek() == Some(&Token::CloseParens) {
        assert_eq!(tokens.next().unwrap(), Token::CloseParens);
        return values;
    }

    loop {
        let (value, token) = parse_value(tokens, reqs);

        values.push(value);

        if Token::CloseParens == token {
            break;
        } else {
            assert_eq!(Token::Comma, token, "{values:?}");
        }
    }

    values
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
    requirements: HashSet<String>,
    block: Block,
}

#[derive(Debug, Clone)]
struct Block {
    variables: HashMap<String, TypedVariable>,
    statements: Vec<TypedStatement>,
}

#[derive(Debug, Clone)]
struct TypedVariable {
    vartype: EmptyType,
    // this is known at declaration so it is not nessicary for this to be here
    // but there is no real reason to move it to declaration
    mutable: bool,
    copied: CopyState,
}

#[derive(Debug, Clone)]
enum CopyState {
    Unused,
    UsedOnce,
    UsedMulti,
}

impl CopyState {
    fn inc(&mut self) {
        *self = match self {
            CopyState::Unused => CopyState::UsedOnce,
            CopyState::UsedOnce => CopyState::UsedMulti,
            CopyState::UsedMulti => CopyState::UsedMulti,
        }
    }
}

#[derive(Debug, Clone)]
enum TypedStatement {
    Declaration {
        // _mutable: bool,
        // copied: bool,
        variable: String,
        value: TypedValue,
    },
    Assignment {
        variable: String,
        value: TypedValue,
    },

    While {
        block: Block,
        value: TypedValue,
    },

    If {
        block: Block,
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
    FunctionCall { name: String, args: Vec<TypedValue> },
    Operation(TypedOperation),
}
#[derive(Debug, Clone)]
enum TypedOperation {
    IntAdd(Box<TypedValue>, Box<TypedValue>),
    IntSub(Box<TypedValue>, u8),
    IntEq(Box<TypedValue>, Box<TypedValue>),
    BoolNot(Box<TypedValue>),
}

// TODO this does not work at all
fn type_ast(mut ast: AST) -> TypedAST {
    let mut sigs = libf_func_sigs();

    for func in ast.functions.iter() {
        assert!(sigs.insert(func.name.clone(), func.sig.clone()).is_none());
    }

    // TODO make this detect recursion

    let mut functions = ast
        .functions
        .into_iter()
        .map(|x| type_function(x, &sigs))
        .collect();

    TypedAST {
        imports: ast.imports,
        functions,
    }
}

fn libf_func_sigs() -> HashMap<String, FunctionSig> {
    HashMap::from_iter(
        [
            (
                String::from("list"),
                FunctionSig {
                    args: Vec::new(),
                    ret: EmptyType::new_list(),
                },
            ),
            (
                String::from("push"),
                FunctionSig {
                    args: vec![EmptyType::new_list(), EmptyType::Int(Empty)],
                    ret: EmptyType::Uninit,
                },
            ),
            // (
            //     String::from("pop"),
            //     FunctionSig {
            //         args: vec![EmptyType::Int(Empty)],
            //         ret: EmptyType::Uninit,
            //     },
            // ),
            (
                String::from("index"),
                FunctionSig {
                    args: vec![EmptyType::new_list(), EmptyType::Int(Empty)],
                    ret: EmptyType::Int(Empty),
                },
            ),
            (
                String::from("index_set"),
                FunctionSig {
                    args: vec![
                        EmptyType::new_list(),
                        EmptyType::Int(Empty),
                        EmptyType::Int(Empty),
                    ],
                    ret: EmptyType::Uninit,
                },
            ),
            (
                String::from("input"),
                FunctionSig {
                    args: vec![],
                    ret: EmptyType::Int(Empty),
                },
            ),
            (
                String::from("output"),
                FunctionSig {
                    args: vec![EmptyType::Int(Empty)],
                    ret: EmptyType::Uninit,
                },
            ),
        ]
        .iter()
        .cloned(),
    )
}

fn type_function(func: Function, sigs: &HashMap<String, FunctionSig>) -> TypedFunction {
    TypedFunction {
        name: func.name,
        sig: func.sig,
        block: type_statements(func.statements, sigs, &mut Vec::new()),
        requirements: func.requirements,
    }
}

fn type_statements(
    statements: Vec<Statement>,
    sigs: &HashMap<String, FunctionSig>,
    var_types: &mut Vec<HashMap<String, TypedVariable>>,
) -> Block {
    var_types.push(HashMap::new());

    let statements = statements
        .into_iter()
        .map(|x| type_statement(x, sigs, var_types))
        .collect();

    Block {
        variables: var_types.pop().unwrap(),
        statements,
    }
}

fn type_statement(
    statement: Statement,
    sigs: &HashMap<String, FunctionSig>,
    var_types: &mut Vec<HashMap<String, TypedVariable>>,
) -> TypedStatement {
    match statement {
        Statement::Declaration {
            variable,
            value,
            mutable,
        } => {
            let value = type_value(value, sigs, var_types);

            // todo find a way to

            let TypedValue {
                val_type,
                val_source: _,
            } = &value;

            var_types.last_mut().unwrap().insert(
                variable.clone(),
                TypedVariable {
                    vartype: val_type.clone(),
                    mutable,
                    copied: CopyState::Unused,
                },
            );

            TypedStatement::Declaration { variable, value }
        }
        Statement::Assignment { variable, value } => TypedStatement::Assignment {
            variable,
            value: type_value(value, sigs, var_types),
        },

        Statement::If { statements, value } => TypedStatement::If {
            block: type_statements(statements, sigs, var_types),
            value: type_value(value, sigs, var_types),
        },

        Statement::While { statements, value } => TypedStatement::While {
            block: type_statements(statements, sigs, var_types),
            value: type_value(value, sigs, var_types),
        },
        Statement::Unit { value } => TypedStatement::Unit {
            value: type_value(value, sigs, var_types),
        },
    }
}

fn type_value(
    value: Value,
    sigs: &HashMap<String, FunctionSig>,
    var_types: &mut Vec<HashMap<String, TypedVariable>>,
) -> TypedValue {
    match value.val_source {
        ValueSource::Constant(literal) => TypedValue {
            val_type: match literal {
                Literal::Int(_) => EmptyType::new_int(),
                Literal::Bool(_) => EmptyType::new_bool(),
            },
            val_source: TypedValueSource::Constant(literal),
        },

        // TODO make this work by making copy a count
        // 0 - unused
        // 1 - used once, no copy needed
        // 2- used many, copy needed
        ValueSource::Variable { name } => {
            let val = find_var(var_types, &name);

            val.copied.inc();

            TypedValue {
                val_type: val.vartype.clone(),
                val_source: TypedValueSource::Variable { name },
            }
        }
        ValueSource::FunctionCall { name, args } => {
            let sig = sigs.get(&name).unwrap();

            let typed_args: Vec<_> = args
                .into_iter()
                .map(|x| type_value(x, sigs, var_types))
                .collect();

            assert_eq!(typed_args.len(), sig.args.len());

            assert!(
                typed_args
                    .iter()
                    .zip(sig.args.iter())
                    .all(|(val, arg_type)| &val.val_type == arg_type)
            );

            TypedValue {
                val_type: sig.ret.clone(),
                val_source: TypedValueSource::FunctionCall {
                    name,
                    args: typed_args,
                },
            }
        }
        ValueSource::Operation(operation) => match operation {
            Operation::IntAdd(value, value1) => {
                let value = type_value(*value, sigs, var_types);
                let value1 = type_value(*value1, sigs, var_types);

                assert_eq!(value.val_type, EmptyType::new_int());
                assert_eq!(value1.val_type, EmptyType::new_int());

                TypedValue {
                    val_type: EmptyType::new_int(),
                    val_source: TypedValueSource::Operation(TypedOperation::IntAdd(
                        Box::new(value),
                        Box::new(value1),
                    )),
                }
            }
            Operation::IntSub(value, val) => {
                let value = type_value(*value, sigs, var_types);

                assert_eq!(value.val_type, EmptyType::new_int());

                TypedValue {
                    val_type: EmptyType::new_int(),
                    val_source: TypedValueSource::Operation(TypedOperation::IntSub(
                        Box::new(value),
                        val,
                    )),
                }
            }
            Operation::IntEq(value, value1) => {
                let value = type_value(*value, sigs, var_types);
                let value1 = type_value(*value1, sigs, var_types);

                assert_eq!(value.val_type, EmptyType::new_int());
                assert_eq!(value1.val_type, EmptyType::new_int());

                TypedValue {
                    val_type: EmptyType::new_bool(),
                    val_source: TypedValueSource::Operation(TypedOperation::IntEq(
                        Box::new(value),
                        Box::new(value1),
                    )),
                }
            }
            Operation::BoolNot(value) => {
                let value = type_value(*value, sigs, var_types);

                assert_eq!(value.val_type, EmptyType::new_bool());

                TypedValue {
                    val_type: EmptyType::new_bool(),
                    val_source: TypedValueSource::Operation(TypedOperation::BoolNot(Box::new(
                        value,
                    ))),
                }
            }
        },
    }
}

// TODO this will panic if the var_name id not found
fn find_var<'a>(
    var_types: &'a mut [HashMap<String, TypedVariable>],
    name: &str,
) -> &'a mut TypedVariable {
    var_types
        .iter_mut()
        .filter_map(|x| x.get_mut(name))
        .next()
        .unwrap()
}

// TODO this does not handle recursion or indirect recursion
fn ast_to_bfasm(ast: TypedAST) -> Vec<crate::bfasm::Instruction> {
    let main_fn = ast
        .functions
        .into_iter()
        .find(|x| x.name == "main")
        .unwrap();

    // TODO compiler mains dependences before compiling it and so on
    // or this dependency initalization could be done by ast_fn_to_bfasm

    assert_eq!(
        main_fn.sig,
        FunctionSig {
            args: Vec::new(),
            ret: EmptyType::Uninit
        }
    );

    block_to_bfasm(main_fn.block, &mut HashMap::new(), 0)
}

fn block_to_bfasm(
    block: Block,
    vars: &mut HashMap<String, (EmptyType, usize)>,
    mut offset: usize,
) -> Vec<crate::bfasm::Instruction> {
    use crate::bfasm::{Instruction, InstructionVariant, PrimativeType};

    let mut instructions = Vec::new();

    for statement in block.statements {
        match statement {
            TypedStatement::Declaration {
                variable,
                value:
                    value @ TypedValue {
                        val_type: EmptyType::Bool(_) | EmptyType::Int(_),
                        val_source: _,
                    },
            } => {
                instructions.append(&mut value_to_bfasm(&value, &vars, offset));

                assert!(
                    vars.insert(variable, (value.val_type, offset))
                        .is_none()
                );

                // inc offset to indicate this space is filled and add and extra 2 spaces to enable
                // cheap copying
                offset += 3;
            }

            // this does not support list copying e.g.
            // let x = list();
            // let y = x;
            TypedStatement::Declaration {
                variable,
                value:
                    TypedValue {
                        val_type: EmptyType::List(_),
                        val_source: TypedValueSource::FunctionCall { name, args: _ },
                    },
            } => {
                assert_eq!(name, "list");

                assert!(
                    vars.insert(
                        variable,
                        (EmptyType::List(crate::bfasm::EmptyVec(Empty)), offset + 2)
                    )
                    .is_none()
                );

                instructions.push(Instruction::new(offset + 2, InstructionVariant::ListNew));
                offset += 5; // _ _ list _ _
            }

            TypedStatement::Declaration {
                variable,
                value:
                    value @ TypedValue {
                        val_type: EmptyType::List(_),
                        val_source: _,
                    },
            } => unimplemented!(),

            TypedStatement::Declaration {
                variable,
                value:
                    value @ TypedValue {
                        val_type: EmptyType::Uninit,
                        val_source: _,
                    },
            } => {
                unreachable!()
            }

            TypedStatement::Assignment { variable, value } => {
                instructions.append(&mut value_to_bfasm(&value, &vars, offset));

                let (var_type, pos) = vars.get(&variable).unwrap();

                match var_type {
                    PrimativeType::Int(_) => {
                        instructions.push(Instruction::new(*pos, InstructionVariant::IntRemove));
                        instructions
                            .push(Instruction::new(offset, InstructionVariant::IntMove(*pos)));
                    }
                    PrimativeType::Bool(_) => {
                        instructions.push(Instruction::new(*pos, InstructionVariant::BoolRemove));
                        instructions
                            .push(Instruction::new(offset, InstructionVariant::BoolMove(*pos)));
                    }
                    PrimativeType::Uninit => unreachable!(),
                    PrimativeType::List(_) => todo!(),
                }
            }

            TypedStatement::While { block, value } => {
                let mut value_instructs = value_to_bfasm(&value, &vars, offset);

                instructions.append(&mut (value_instructs.clone()));

                let mut while_instructs = Vec::new();

                while_instructs.push(Instruction::new(offset, InstructionVariant::BoolRemove));

                while_instructs.append(&mut block_to_bfasm(block, vars, offset));

                // this code assumes that declarations are added in increasing offsets
                for (name, (t, pos)) in vars.extract_if(|_, x| x.1 >= offset) {
                    match t {
                        PrimativeType::Uninit => unreachable!(),
                        PrimativeType::Int(_) => while_instructs
                            .push(Instruction::new(pos, InstructionVariant::IntRemove)),
                        PrimativeType::Bool(_) => while_instructs
                            .push(Instruction::new(pos, InstructionVariant::BoolRemove)),
                        PrimativeType::List(_) => unimplemented!(),
                    }
                }

                while_instructs.append(&mut value_instructs);

                instructions.push(Instruction::new(
                    offset,
                    InstructionVariant::While(while_instructs),
                ));

                instructions.push(Instruction::new(offset, InstructionVariant::BoolRemove));
            }
            TypedStatement::If { block, value } => {
                instructions.append(&mut value_to_bfasm(&value, &vars, offset));

                let mut while_instructs = Vec::new();

                while_instructs.push(Instruction::new(offset, InstructionVariant::BoolRemove));

                while_instructs.append(&mut block_to_bfasm(block, vars, offset));

                // this code assumes that declarations are added in increasing offsets
                for (name, (t, pos)) in vars.extract_if(|_, x| x.1 >= offset) {
                    match t {
                        PrimativeType::Uninit => unreachable!(),
                        PrimativeType::Int(_) => {
                            while_instructs.push(Instruction::new(pos, InstructionVariant::IntRemove))
                        }
                        PrimativeType::Bool(_) => {
                            while_instructs.push(Instruction::new(pos, InstructionVariant::BoolRemove))
                        }
                        PrimativeType::List(_) => unimplemented!(),
                    }
                }


                while_instructs.push(Instruction::new(offset, InstructionVariant::BoolSet(false)));

                instructions.push(Instruction::new(
                    offset,
                    InstructionVariant::While(while_instructs),
                ));

                instructions.push(Instruction::new(offset, InstructionVariant::BoolRemove));
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
) -> Vec<crate::bfasm::Instruction> {
    use crate::bfasm::{Instruction, InstructionVariant, PrimativeType};

    match &value.val_source {
        TypedValueSource::Constant(Literal::Int(x)) => {
            vec![Instruction::new(offset, InstructionVariant::IntSet(*x))]
        }

        TypedValueSource::Constant(Literal::Bool(x)) => {
            vec![Instruction::new(offset, InstructionVariant::BoolSet(*x))]
        }

        TypedValueSource::Variable { name } => {
            let x = vars.get(name).unwrap();

            let index = x.1;

            match x.0 {
                PrimativeType::Int(_) => {
                    vec![
                        Instruction::new(index, InstructionVariant::IntCopy),
                        Instruction::new(index + 1, InstructionVariant::IntMove(index)),
                        Instruction::new(index + 2, InstructionVariant::IntMove(offset)),
                    ]
                }
                PrimativeType::Bool(_) => {
                    vec![
                        Instruction::new(index, InstructionVariant::BoolCopy),
                        Instruction::new(index + 1, InstructionVariant::BoolMove(index)),
                        Instruction::new(index + 2, InstructionVariant::BoolMove(offset)),
                    ]
                }
                PrimativeType::Uninit => unreachable!(),
                PrimativeType::List(_) => todo!(),
            }
        }
        TypedValueSource::FunctionCall { name, args } => {
            match name.as_str() {
                // this will never be called because list is never created outside of a call to
                // list() and a call to list is never used to do anything besides create a list
                "list" => {
                    unimplemented!()
                }
                "push" => {
                    // this does not handle list().push(x); as the val source is not a variable
                    let [
                        TypedValue {
                            val_type: EmptyType::List(_),
                            val_source: TypedValueSource::Variable { name },
                        },
                        int_value @ TypedValue {
                            val_type: EmptyType::Int(_),
                            val_source: _,
                        },
                    ] = &args[..]
                    else {
                        panic!();
                    };

                    let Some((EmptyType::List(_), list_pos)) = vars.get(name) else {
                        panic!()
                    };

                    let mut instructs = Vec::new();

                    instructs.append(&mut value_to_bfasm(&int_value, vars, offset));

                    // this should be safe bc lists should have 2 empty cells on either side
                    instructs.push(Instruction::new(
                        offset,
                        InstructionVariant::IntMove(*list_pos + 1),
                    ));

                    // make space for growth
                    let mut to_move = vars
                        .iter()
                        .map(|x| x.1)
                        .filter(|x| x.1 > *list_pos)
                        .collect::<Vec<_>>();

                    // TODO we should sort into reverse instead of reversing the iter
                    to_move.sort_by_key(|x| x.1);
                    to_move.iter().rev().for_each(|x| match x.0 {
                        PrimativeType::Uninit => unimplemented!(),
                        PrimativeType::Int(_) => instructs
                            .push(Instruction::new(x.1, InstructionVariant::IntMove(x.1 + 2))),
                        PrimativeType::Bool(_) => instructs
                            .push(Instruction::new(x.1, InstructionVariant::BoolMove(x.1 + 2))),
                        PrimativeType::List(_) => todo!(),
                    });

                    instructs.push(Instruction::new(*list_pos, InstructionVariant::ListPush));

                    instructs
                }

                "input" => {
                    vec![Instruction::new(offset, InstructionVariant::Input)]
                }

                "output" => {
                    let [
                        int_value @ TypedValue {
                            val_type: EmptyType::Int(_),
                            val_source: _,
                        },
                    ] = &args[..]
                    else {
                        panic!();
                    };

                    let mut instructs = value_to_bfasm(int_value, vars, offset);

                    instructs.push(Instruction::new(offset, InstructionVariant::Output));
                    instructs.push(Instruction::new(offset, InstructionVariant::IntRemove));

                    instructs
                }

                "pop" => todo!(),

                "index" => {
                    let [
                        TypedValue {
                            val_type: EmptyType::List(_),
                            val_source: TypedValueSource::Variable { name },
                        },
                        int_value @ TypedValue {
                            val_type: EmptyType::Int(_),
                            val_source: _,
                        },
                    ] = &args[..]
                    else {
                        panic!();
                    };

                    let Some((EmptyType::List(_), list_pos)) = vars.get(name) else {
                        panic!()
                    };

                    let mut instructs = Vec::new();

                    instructs.append(&mut value_to_bfasm(&int_value, vars, offset));

                    // this should be safe bc lists should have 2 empty cells on either side
                    instructs.push(Instruction::new(
                        offset,
                        InstructionVariant::IntMove(*list_pos - 1),
                    ));

                    instructs.push(Instruction::new(
                        *list_pos - 2,
                        InstructionVariant::ListIndex,
                    ));

                    instructs.push(Instruction::new(
                        *list_pos - 1,
                        InstructionVariant::IntMove(offset),
                    ));

                    instructs
                }
                "index_set" => {
                    let [
                        TypedValue {
                            val_type: EmptyType::List(_),
                            val_source: TypedValueSource::Variable { name },
                        },
                        idx_val @ TypedValue {
                            val_type: EmptyType::Int(_),
                            val_source: _,
                        },
                        int_val @ TypedValue {
                            val_type: EmptyType::Int(_),
                            val_source: _,
                        },
                    ] = &args[..]
                    else {
                        panic!()
                    };

                    let Some((EmptyType::List(_), list_pos)) = vars.get(name) else {
                        panic!()
                    };

                    // [WARN] this assumes that no other function call will use the list in between
                    let mut instructs = value_to_bfasm(int_val, vars, offset);
                    instructs.push(Instruction::new(
                        offset,
                        InstructionVariant::IntMove(*list_pos - 2),
                    ));

                    instructs.append(&mut value_to_bfasm(idx_val, vars, offset));
                    instructs.push(Instruction::new(
                        offset,
                        InstructionVariant::IntMove(*list_pos - 1),
                    ));

                    instructs.push(Instruction::new(
                        *list_pos - 2,
                        InstructionVariant::ListIndexSet,
                    ));

                    instructs
                }
                _ => todo!("{name},"),
            }
        }
        TypedValueSource::Operation(op) => {
            let mut instructs = Vec::new();
            match op {
                TypedOperation::IntAdd(value1, value2) => {
                    instructs.append(&mut value_to_bfasm(&value1, vars, offset));
                    instructs.append(&mut value_to_bfasm(&value2, vars, offset + 1));

                    instructs.push(Instruction::new(offset, InstructionVariant::IntAdd));
                }

                TypedOperation::IntSub(value, num) => {
                    instructs.append(&mut value_to_bfasm(&value, vars, offset));
                    (0..*num).for_each(|_| {
                        instructs.push(Instruction::new(offset, InstructionVariant::UnsafeSub))
                    });
                }

                TypedOperation::IntEq(value1, value2) => {
                    instructs.append(&mut value_to_bfasm(&value1, vars, offset));
                    instructs.append(&mut value_to_bfasm(&value2, vars, offset + 2));

                    instructs.push(Instruction::new(offset, InstructionVariant::IntEq));
                }

                TypedOperation::BoolNot(value) => {
                    instructs.append(&mut value_to_bfasm(&value, vars, offset));

                    instructs.push(Instruction::new(offset, InstructionVariant::BoolNot));
                }
            }

            instructs
        }
    }
}

pub fn main() {
    let program = std::fs::read_to_string("./program/src/main.rs").unwrap();

    let tokens = tokenize(&program);

    // dbg!(&tokens);

    let ast = tokens_to_ast(tokens);

    // dbg!(&ast);

    let typed_ast = type_ast(ast);

    // dbg!(&typed_ast);

    let instructs = ast_to_bfasm(typed_ast);

    // dbg!(&instructs);

    let mut compiler = DebugComplier::default();

    compiler.push_input_str(",[->+<]>.\0a");
    // compiler.push_input_str("+++[->+<]\0");

    compiler.exec_instructs(&instructs);

    dbg!(compiler.to_str());

    let output = compiler.get_output().iter().fold(String::new(), |mut s, x|{ s.push(*x as char); s});

    dbg!(output);

}

#[cfg(test)]
mod test {}
