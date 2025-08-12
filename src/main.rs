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
    Int(usize),
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
    statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
enum Statement {
    Declaration {
        mutable: bool,
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

                statements.push(Statement::Declaration {
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

fn ast_to_bfasm(ast: AST) -> Vec<bfasm::Instruction> {
    assert_eq!(&ast.imports, &[String::from("libf")]);

    todo!()
}

fn ast_fn_to_bfasm(ast: FunctionDec) {}

fn main() {
    let program = std::fs::read_to_string("./program/src/main.rs").unwrap();

    let tokens = tokenize(&program);

    dbg!(&tokens);

    let ast = tokens_to_ast(tokens);

    dbg!(&ast);
}

#[cfg(test)]
mod test {}
