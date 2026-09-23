import math


def parse_nmea_coordinate(coord_str, direction_str):
    """
    Преобразует координату из формата NMEA (ГГММ.МММММ) в десятичные градусы.
    
    :param coord_str: координата в формате ГГММ.МММММ
    :param direction_str: направление (N/S для широты, E/W для долготы)
    :return: координата в десятичных градусах
    """
    if not coord_str or not direction_str:
        return None
    
    try:
        # Парсим строку в число
        coord_float = float(coord_str)
        
        # Извлекаем градусы (целая часть / 100)
        degrees = int(coord_float // 100)
        
        # Извлекаем минуты (остаток от деления на 100)
        minutes = coord_float % 100
        
        # Преобразуем в десятичные градусы
        decimal_degrees = degrees + minutes / 60.0
        
        # Учитываем направление (S и W - отрицательные)
        if direction_str in ['S', 'W']:
            decimal_degrees = -decimal_degrees
        
        return decimal_degrees
    except (ValueError, IndexError):
        return None


def extract_lat_lon_from_pos(pos_string):
    """
    Извлекает широту и долготу из NMEA строки pos.
    
    Формат строки: "GNRMC,time,status,lat,NS,lon,EW,..."
    Индексы:        0     1     2      3   4   5   6
    
    :param pos_string: строка NMEA
    :return: (lat, lon) в десятичных градусах или (None, None)
    """
    if not pos_string:
        return None, None
    
    try:
        # Убираем кавычки и пробелы, разбиваем по запятой
        pos_clean = pos_string.strip().strip('"')
        parts = pos_clean.split(',')
        
        if len(parts) < 7:
            return None, None
        
        # Извлекаем координаты и направления
        lat_str = parts[3].strip() if len(parts) > 3 else ''
        ns_str = parts[4].strip() if len(parts) > 4 else ''
        lon_str = parts[5].strip() if len(parts) > 5 else ''
        ew_str = parts[6].strip() if len(parts) > 6 else ''
        
        # Преобразуем в десятичные градусы
        lat = parse_nmea_coordinate(lat_str, ns_str)
        lon = parse_nmea_coordinate(lon_str, ew_str)
        
        return lat, lon
    except Exception as e:
        print(f"️  Ошибка парсинга NMEA координат: {e}")
        return None, None


def gps_to_meters(lat, lon, ref_lat, ref_lon):
    """
    Локальная плоская проекция (Equirectangular approximation).
    Корректна для дистанций до нескольких километров.
    """
    R = 6378137.0
    
    # Переводим в радианы
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    
    # Ось X: расстояние по параллели (с учетом cos(shiroty))
    x = R * (lon_rad - ref_lon_rad) * math.cos(ref_lat_rad)
    
    # Ось Z: расстояние по меридиану
    z = R * (lat_rad - ref_lat_rad)
    
    return x, z