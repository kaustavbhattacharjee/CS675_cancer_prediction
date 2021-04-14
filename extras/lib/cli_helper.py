import sys


def print_help_message():
    help_message = "\nUsage:  python main.py "
    help_message = help_message + "<TRAIN_DATA_FILE_PATH> <TEST_DATA_FILE_PATH> <TRAINLABEL_FILE_PATH>"
    help_message = help_message + " [OPTIONS]\n\nA feature selector and classifier package\n\nOptions:\n"
    help_message = help_message + "  -k, --dims\t\tNumber of dimensions to reduce to finally. Default is 15\n"
    help_message = help_message + "  -c, --combinations\t[snr pearson mi silhouette chi2]. "
    help_message = help_message + "Space separated correlation metrics. Default is a combination of mi and chi2."

    print(help_message)
    sys.exit(0)


def sanity_check():
    if (len(sys.argv) < 4):
        print_help_message()

    def _supported_option(i, check=("k", "c")):
        if not check:
            print("Argument not known. Ignoring the argument.")
            print_help_message()

        cond2 = False
        if "k" in check:
            cond2 = cond2 or sys.argv[i] != "-k" or sys.argv[i] != "--dims"
        if "c" in check:
            cond2 = cond2 or sys.argv[i] != '-c' or sys.argv[i] != "--combinations"
        if sys.argv[i][1] == "-" and cond2:
            print("Option not supported")
            print_help_message()

    def _support_combination(methods):
        for method in methods:
            if method not in ["snr", "pearson", "mi", "silhouette", "chi2"]:
                print("Correlation method not supported.")
                print_help_message()

    def _value_error():
        print("Values for option missing")
        print_help_message()

    def _k_error():
        print("Number of dimensions to reduce to finally should be an integer.")
        print_help_message()

    k = 15
    combination = ("mi", "chi2")

    if len(sys.argv) > 4:
        if sys.argv[4] == "-k" or sys.argv[4] == "--dims":
            if len(sys.argv) < 6:
                print("Values for option missing")
                print_help_message()
            try:
                k = int(sys.argv[5])
            except ValueError:
                _k_error()
            if len(sys.argv) > 6:
                if sys.argv[6] == "-c" or sys.argv[6] == "--combinations":
                    if len(sys.argv) < 8:
                        print("Values for option missing")
                        print_help_message()
                    combination = tuple(sys.argv[7:])
                else:
                    _supported_option(6, check=())
        elif sys.argv[4] == "-c" or sys.argv[4] == "--combinations":
            combs = []
            for i in range(5, len(sys.argv)):
                if sys.argv[i] == "-k" or sys.argv[i] == "--dims":
                    try:
                        k = int(sys.argv[i + 1])
                    except ValueError:
                        _k_error()
                    except IndexError:
                        _value_error()
                    break
                combs.append(sys.argv[i])
            combination = tuple(combs)

    _support_combination(combination)

    return k, combination
