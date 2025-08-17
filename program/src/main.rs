mod libf;

use libf::*;

fn main() {
    let mut vec = list();

    let mut x = 0;

    while !(x == 6) {
        vec.push(x);

        x += 1;
    }
}
