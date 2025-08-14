mod libf;

fn main() {
    // let mut x = 1;

    // let mut y = x;

    // y = y + 1;

    // x = y;

    // let mut x = 0;

    // let y = x == 1;

    // x = x + 2;

    // let z = x == 2;

    let mut x = 0;

    let mut y = (x == 6);

    while !y {
        x = x + 1;
        y = x == 6;
    }
}
