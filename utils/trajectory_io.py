# utils/trajectory_io.py
import csv
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

def save_trajectory_csv(filepath: str, x: List[float], z: List[float], 
                        metadata: Optional[Dict] = None):
    """Сохраняет траекторию в CSV: frame,x,z"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'x', 'z'])
        for i, (xi, zi) in enumerate(zip(x, z)):
            writer.writerow([i, xi, zi])
        if metadata:
            writer.writerow([])
            for k, v in metadata.items():
                writer.writerow([f'# {k}', v])
    print(f"✅ Saved: {filepath}")

def load_trajectory_csv(filepath: str) -> Tuple[List[float], List[float], Dict]:
    """Загружает траекторию из CSV, возвращает (x, z, metadata)"""
    x, z = [], []
    metadata = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            if row[0].startswith('#'):
                if len(row) >= 2:
                    metadata[row[0][2:]] = row[1]
                continue
            if row[0] == 'frame': continue  # заголовок
            try:
                x.append(float(row[1]))
                z.append(float(row[2]))
            except (IndexError, ValueError):
                continue
    return x, z, metadata

def save_trajectory_json(filepath: str, x: List[float], z: List[float], 
                         metadata: Optional[Dict] = None):
    """Альтернативный формат: JSON"""
    data = {
        'x': x,
        'z': z,
        'metadata': metadata or {}
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved: {filepath}")

def load_trajectory_json(filepath: str) -> Tuple[List[float], List[float], Dict]:
    data = json.loads(Path(filepath).read_text(encoding='utf-8'))
    return data['x'], data['z'], data.get('metadata', {})