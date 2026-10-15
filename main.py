
from pathlib import Path
import math
from collections import deque

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent


# ==================================================
# 1. ЗАГРУЗКА ПРОЦЕССОВ
# ==================================================

def load_processes(filename):
    processes = []

    with open(BASE_DIR / filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            pid, arrival, burst, priority = line.split()

            process = {
                "pid": pid,
                "arrival": int(arrival),
                "burst": int(burst),
                "priority": int(priority),
            }

            if process["arrival"] < 0 or process["burst"] <= 0:
                raise ValueError(f"Некорректные данные: {line}")

            processes.append(process)

    if not processes:
        raise ValueError(f"Файл {filename} пуст")

    if len({p["pid"] for p in processes}) != len(processes):
        raise ValueError("PID процессов должны быть уникальными")

    return sorted(processes, key=lambda p: (p["arrival"], p["pid"]))


# ==================================================
# 2. СИМУЛЯТОР ПЛАНИРОВАНИЯ CPU
# ==================================================

def simulate(processes, algorithm, **params):
    """
    Поддерживаемые алгоритмы:
    fcfs, sjf, srtf, priority, rr.

    Интервал:
    {"start": 0, "end": 3, "pid": "P1"}

    Для RR: quantum=3.
    Для Priority: aging=True, aging_interval=5.
    Меньшее значение priority означает более высокий приоритет.
    """

    if algorithm not in {"fcfs", "sjf", "srtf", "priority", "rr"}:
        raise ValueError(f"Неизвестный алгоритм: {algorithm}")

    quantum = params.get("quantum", 2)
    aging = params.get("aging", False)
    aging_interval = params.get("aging_interval", 5)

    if algorithm == "rr" and quantum <= 0:
        raise ValueError("Квант должен быть положительным")

    if aging_interval <= 0:
        raise ValueError("aging_interval должен быть положительным")

    jobs = {
        p["pid"]: {
            **p,
            "remaining": p["burst"],
            "first_start": None,
            "completion": None,
        }
        for p in processes
    }

    arrivals = sorted(
        processes,
        key=lambda p: (p["arrival"], p["pid"])
    )

    ready = deque()
    current = None
    pending_requeue = None
    quantum_used = 0
    arrival_index = 0
    completed = 0
    time = 0
    timeline = []

    while completed < len(jobs):

        # Сначала добавляем прибывшие процессы.
        while (
            arrival_index < len(arrivals)
            and arrivals[arrival_index]["arrival"] <= time
        ):
            ready.append(arrivals[arrival_index]["pid"])
            arrival_index += 1

        # При RR новые процессы добавляются раньше
        # процесса, чей квант закончился в тот же момент.
        if pending_requeue is not None:
            ready.append(pending_requeue)
            pending_requeue = None

        if algorithm in {"fcfs", "sjf"}:
            if current is None and ready:
                if algorithm == "fcfs":
                    current = ready.popleft()
                else:
                    current = min(
                        ready,
                        key=lambda pid: (
                            jobs[pid]["burst"],
                            jobs[pid]["arrival"],
                            pid,
                        ),
                    )
                    ready.remove(current)

        elif algorithm == "rr":
            if current is None and ready:
                current = ready.popleft()
                quantum_used = 0

        else:
            candidates = list(ready)

            if current is not None:
                candidates.append(current)

            if candidates:

                def scheduling_key(pid):
                    job = jobs[pid]

                    if algorithm == "srtf":
                        return (
                            job["remaining"],
                            job["arrival"],
                            pid,
                        )

                    effective_priority = job["priority"]

                    if aging:
                        executed = job["burst"] - job["remaining"]
                        waiting = max(
                            0,
                            time - job["arrival"] - executed,
                        )

                        effective_priority -= (
                            waiting // aging_interval
                        )

                    return (
                        effective_priority,
                        job["arrival"],
                        pid,
                    )

                selected = min(candidates, key=scheduling_key)

                if selected != current:
                    if current is not None:
                        ready.append(current)

                    ready.remove(selected)
                    current = selected

        # Если процессор свободен, фиксируем простой.
        if current is None:
            timeline.append((time, time + 1, None))
            time += 1
            continue

        job = jobs[current]

        if job["first_start"] is None:
            job["first_start"] = time

        timeline.append((time, time + 1, current))

        job["remaining"] -= 1
        time += 1

        if algorithm == "rr":
            quantum_used += 1

        if job["remaining"] == 0:
            job["completion"] = time
            completed += 1
            current = None
            quantum_used = 0

        elif algorithm == "rr" and quantum_used >= quantum:
            pending_requeue = current
            current = None
            quantum_used = 0

    # Объединяем соседние интервалы одного процесса.
    intervals = []

    for start, end, pid in timeline:
        if (
            intervals
            and intervals[-1]["pid"] == pid
            and intervals[-1]["end"] == start
        ):
            intervals[-1]["end"] = end
        else:
            intervals.append({
                "start": start,
                "end": end,
                "pid": pid,
            })

    return intervals


# ==================================================
# 3. РАСЧЁТ МЕТРИК
# ==================================================

def calculate_metrics(processes, intervals):
    jobs = {p["pid"]: p for p in processes}

    first_start = {}
    completion = {}

    for interval in intervals:
        pid = interval["pid"]

        if pid is None:
            continue

        first_start.setdefault(pid, interval["start"])
        completion[pid] = interval["end"]

    waiting_times = []
    turnaround_times = []
    response_times = []

    for pid, process in jobs.items():
        turnaround = completion[pid] - process["arrival"]
        waiting = turnaround - process["burst"]
        response = first_start[pid] - process["arrival"]

        waiting_times.append(waiting)
        turnaround_times.append(turnaround)
        response_times.append(response)

    start_time = min(p["arrival"] for p in processes)
    end_time = max(completion.values())
    elapsed = end_time - start_time

    busy_time = sum(p["burst"] for p in processes)

    utilization = (
        100 * busy_time / elapsed
        if elapsed > 0
        else 0
    )

    # Считаем переходы от одного процесса к другому.
    # Простой CPU не считается переключением между процессами.
    switches = 0
    previous_pid = None

    for interval in intervals:
        pid = interval["pid"]

        if pid is not None:
            if previous_pid is not None and pid != previous_pid:
                switches += 1

            previous_pid = pid

    return {
        "waiting": sum(waiting_times) / len(waiting_times),
        "turnaround": sum(turnaround_times) / len(turnaround_times),
        "response": sum(response_times) / len(response_times),
        "utilization": utilization,
        "switches": switches,
    }


def print_gantt(intervals):
    print("\nДиаграмма Ганта:")

    for interval in intervals:
        pid = interval["pid"] or "IDLE"
        print(
            f"[{interval['start']}-{interval['end']}: {pid}]",
            end=" "
        )

    print()


# ==================================================
# 4. СРАВНЕНИЕ АЛГОРИТМОВ
# ==================================================

def run_comparison(filename):
    processes = load_processes(filename)

    algorithms = [
        ("FCFS", "fcfs", {}),
        ("SJF", "sjf", {}),
        ("SRTF", "srtf", {}),
        ("Priority", "priority", {}),
        (
            "Priority+Aging",
            "priority",
            {"aging": True, "aging_interval": 5},
        ),
        ("RR(q=3)", "rr", {"quantum": 3}),
    ]

    print("\n" + "=" * 85)
    print(f"НАБОР ДАННЫХ: {filename}")
    print("=" * 85)

    print(
        f"{'Алгоритм':<18}"
        f"{'Ожидание':>11}"
        f"{'Оборот':>11}"
        f"{'Отклик':>11}"
        f"{'CPU %':>11}"
        f"{'Переключ.':>13}"
    )

    results = {}

    for name, algorithm, params in algorithms:
        intervals = simulate(processes, algorithm, **params)
        metrics = calculate_metrics(processes, intervals)

        results[name] = metrics

        print(
            f"{name:<18}"
            f"{metrics['waiting']:>11.2f}"
            f"{metrics['turnaround']:>11.2f}"
            f"{metrics['response']:>11.2f}"
            f"{metrics['utilization']:>10.2f}%"
            f"{metrics['switches']:>13}"
        )

    if filename == "set1.txt":
        expected = {
            "FCFS": (9.25, 15.75),
            "SRTF": (6.5, 13.0),
            "RR(q=3)": (13.5, 20.0),
        }

        print("\nПРОВЕРКА ПО ЭТАЛОНУ")

        for name, (expected_wait, expected_turnaround) in expected.items():
            actual = results[name]

            wait_ok = math.isclose(
                actual["waiting"],
                expected_wait,
                abs_tol=1e-9,
            )

            turnaround_ok = math.isclose(
                actual["turnaround"],
                expected_turnaround,
                abs_tol=1e-9,
            )

            status = (
                "OK"
                if wait_ok and turnaround_ok
                else "НЕ СОВПАЛО"
            )

            print(
                f"{name}: ожидание={actual['waiting']:.2f}, "
                f"оборот={actual['turnaround']:.2f} — {status}"
            )

    return processes, results


# ==================================================
# 5. ГРАФИК ROUND ROBIN
# ==================================================

def plot_rr_quantum(filename="set2.txt"):
    processes = load_processes(filename)

    quantums = list(range(1, 11))
    waiting = []
    switches = []

    for quantum in quantums:
        intervals = simulate(
            processes,
            "rr",
            quantum=quantum,
        )

        metrics = calculate_metrics(processes, intervals)

        waiting.append(metrics["waiting"])
        switches.append(metrics["switches"])

    fig, ax1 = plt.subplots(figsize=(10, 5))

    line1 = ax1.plot(
        quantums,
        waiting,
        marker="o",
        label="Среднее время ожидания",
    )

    ax1.set_xlabel("Квант времени")
    ax1.set_ylabel("Среднее время ожидания")
    ax1.set_xticks(quantums)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    line2 = ax2.plot(
        quantums,
        switches,
        marker="s",
        label="Переключения",
    )

    ax2.set_ylabel("Количество переключений")

    lines = line1 + line2
    labels = [line.get_label() for line in lines]

    ax1.legend(lines, labels, loc="best")

    plt.title("Round Robin: влияние кванта на set2")
    plt.tight_layout()

    plt.savefig(
        BASE_DIR / "rr_quantum.png",
        dpi=150,
    )

    print("\nГрафик сохранён в rr_quantum.png")
    plt.show()


# ==================================================
# 6. ДЕМОНСТРАЦИЯ AGING И ГОЛОДАНИЯ
# ==================================================

def demonstrate_aging():
    # Один длинный процесс с низким приоритетом.
    # Каждую единицу времени прибывает новый короткий
    # процесс с высоким приоритетом.

    processes = [
        {
            "pid": "LONG",
            "arrival": 0,
            "burst": 10,
            "priority": 10,
        }
    ] + [
        {
            "pid": f"H{i}",
            "arrival": i - 1,
            "burst": 2,
            "priority": 1,
        }
        for i in range(1, 21)
    ]

    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ AGING И ГОЛОДАНИЯ")
    print("=" * 70)

    for enabled in (False, True):
        intervals = simulate(
            processes,
            "priority",
            aging=enabled,
            aging_interval=3,
        )

        metrics = calculate_metrics(processes, intervals)

        long_start = next(
            interval["start"]
            for interval in intervals
            if interval["pid"] == "LONG"
        )

        label = "С AGING" if enabled else "БЕЗ AGING"

        print(f"\n{label}")
        print(f"Среднее ожидание: {metrics['waiting']:.2f}")
        print(f"Первый запуск LONG: {long_start}")
        print_gantt(intervals)


# ==================================================
# 7. ЗАПУСК ПРОГРАММЫ
# ==================================================

if __name__ == "__main__":
    for dataset in ("set1.txt", "set2.txt", "set3.txt"):
        run_comparison(dataset)

    demonstrate_aging()
    plot_rr_quantum("set2.txt")
