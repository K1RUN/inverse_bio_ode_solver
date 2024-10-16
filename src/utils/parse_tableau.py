import fractions as fr


def input_butcher_tableau() -> dict:
    """
    NEED TO SPECIFY BUTCHER TABLE FILE IN STDIN (default path is butcher_tables/...)
    :return: dict of lists with floats
    """
    path = 'butcher_tables/'
    filename = input("Enter the name of the Butcher's table: ")
    return parse_butcher_tableau(path + filename)


def parse_butcher_tableau(path: str) -> dict:
    """
    NEED TO SPECIFY BUTCHER TABLE FILE IN PATH
    :return: dict of lists with floats
    """
    with open(path) as file:
        info = file.read().splitlines()

    def is_valid_number(s: str):
        """CHECK IF STRING CONTAINS A VALID NUMBER: POSITIVE OR NEGATIVE RATIONAL (WITH '/' SYMBOL) OR INTEGER"""
        values = s.split('/')
        return (len(values) == 1 or len(values) == 2) and all(num.lstrip('-+').isdigit() for num in values)

    table = {}

    table_size = len(info[0].split()) - 1

    for i, line in enumerate(info):
        """LAST LINE IN INFO STANDS FOR b COEFFICIENTS, b COEFFICIENTS WILL BE ADDED SEPARATELY AFTER THE CYCLE"""
        """https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Implicit_Runge%E2%80%93Kutta_methods"""
        row = line.split()

        if "rank" in row:
            ranks = row[1:]
            table.setdefault('rank', []).extend(list(map(int, ranks)))

        if all(is_valid_number(string) for string in row):
            row = [float(fr.Fraction(number)) for number in row]

            if i < table_size:
                table.setdefault('c_', []).append(row[0])
                table.setdefault('a_', []).append(row[1:])

            elif i == table_size:
                table.setdefault('b_', []).extend(row)

            elif i == table_size + 1:
                table.setdefault('b_star', []).extend(row)

    return table
