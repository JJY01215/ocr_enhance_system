def levenshtein_distance(a: str, b: str) -> int:
    """
    最短編輯距離：計算從 a 變成 b 需要幾次操作（插入/刪除/替換）
    """
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr

    return prev[-1]

def char_accuracy(gt: str, pred: str) -> float:
    """
    字元正確率：
    accuracy = 1 - (edit_distance / len(gt))
    """
    gt2 = gt.strip()
    pred2 = pred.strip()

    dist = levenshtein_distance(gt2, pred2)
    denom = max(len(gt2), 1)

    acc = 1.0 - (dist / denom)
    if acc < 0:
        acc = 0.0
    return acc
