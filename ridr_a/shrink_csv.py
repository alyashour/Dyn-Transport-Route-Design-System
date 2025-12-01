import csv
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python shrink.py <input.csv> <output.csv> <num_rows>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    limit = int(sys.argv[3])  # how many rows (excluding header)

    with open(input_file, newline='', encoding='utf-8') as infile, \
         open(output_file, "w", newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for i, row in enumerate(reader):
            # write the header (i == 0) plus first N rows (i <= limit)
            if i > limit:
                break
            writer.writerow(row)

    print(f"Wrote {limit} rows (plus header) to {output_file}")

if __name__ == "__main__":
    main()
