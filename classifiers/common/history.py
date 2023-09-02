def saveHistory(history, filename):
    f = open(filename, "a")

    for key in history.history.keys():
        
        line_to_write = key + ";"
        
        for val in history.history[key]:
            line_to_write = line_to_write + str(val) + ","

        # Delete last comma
        line_to_write = line_to_write[:-1]

        f.write(line_to_write)
        f.write("\n")

    f.close()