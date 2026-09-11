import sys

def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        parts = line.strip().split()
        if len(parts) != 2:
            break
        opp_r, opp_c = int(parts[0]), int(parts[1])
        valid_count_line = sys.stdin.readline()
        if not valid_count_line:
            break
        valid_count = int(valid_count_line.strip())
        moves = []
        for _ in range(valid_count):
            move_line = sys.stdin.readline()
            if not move_line:
                break
            mr, mc = map(int, move_line.strip().split())
            moves.append((mr, mc))
        if moves:
            best = moves[0]
            sys.stdout.write(f"{best[0]} {best[1]}\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
