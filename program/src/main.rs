mod libf;

use libf::*;

fn main() {
    // let mut vec = list();
    //
    // vec.push(0);
    // vec.push(1);
    // vec.push(2);
    //
    // output(vec.index(2));
    //
    // vec.index_set(2, input());
    // output(vec.index(2));

    let mut instructs = list();
    let mut instructs_len = 0;

    let mut curr_input = input();

    while !(curr_input == 0) {
        instructs.push(curr_input);

        instructs_len += 1;

        curr_input = input();
    }

    let mut instruct_index = 0;

    let mut array = list();
    let mut array_len = 1;

    array.push(0);

    let mut pointer = 0;

    while instruct_index != instructs_len {
        let instruct = instructs.index(instruct_index);

        // > 62
        if instruct == 62 {
            pointer += 1;
            if pointer == array_len {
                array_len += 1;
                array.push(0);
            }
        }

        // < 60
        if instruct == 60 {
            pointer -= 1;
        }

        // + 43
        if instruct == 43 {
            let x = array.index(pointer) + 1;
            array.index_set(pointer, x);
        }

        // - 45
        if instruct == 45 {
            let x = array.index(pointer) - 1;
            array.index_set(pointer, x);
        }

        //         // [ 91
        if instruct == 91 {
            if array.index(pointer) == 0 {
                let mut count = 1;

                instruct_index += 1;

                while count != 0 {
                    let x = instructs.index(instruct_index);

                    if x == '[' as usize {
                        count += 1;
                    }

                    if x == ']' as usize {
                        count -= 1;
                    }

                    instruct_index += 1;
                }

                instruct_index -= 1;
            }
        }
        // ] 93
        if instruct == ']' as usize {
            if array.index(pointer) != 0 {
                let mut count = 1;

                instruct_index -= 1;

                while count != 0 {
                    let x = instructs.index(instruct_index);

                    if x == '[' as usize {
                        count -= 1;
                    }

                    if x == ']' as usize {
                        count += 1;
                    }

                    instruct_index -= 1;
                }

                instruct_index += 1;
            }
        }
        // , 44
        if instruct == 44 {
            array.index_set(pointer, input());
        }

        // .`46
        if instruct == 46 {
            output(array.index(pointer));
        }

        instruct_index += 1;
    }
}
