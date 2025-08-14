mod libf;

fn main() {
    let mut x = 5;

    let mut y = 3;

    let mut b = true;

    if x == 0 {
        b = false;
    }

    if y == 0 {
        b = false;
    }

    while b {
        x = x - 1;
        y = y - 1;

        if x == 0 {
            b = false;
        }

        if y == 0 {
            b = false;
        }
    }

    let z = x + y;
}
