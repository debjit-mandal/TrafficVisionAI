import geopy
from geopy.distance import geodesic


def get_gps_coordinates():
    return (latitude, longitude)


def correlate_sign_with_location(sign, gps_coordinates):
    pass


def main():
    gps_coordinates = get_gps_coordinates()
    sign = "Stop"
    correlate_sign_with_location(sign, gps_coordinates)


if __name__ == "__main__":
    main()
