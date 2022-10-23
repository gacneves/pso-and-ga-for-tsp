import pandas as pd


def read_table_of_distances(fp):
    """
    Open a .csv file of the distances between each city pair

    :param fp: File path
    :return: Data frame of the distances
    """
    df = pd.read_csv(fp, index_col=0)

    # Check if there is a NaN value
    if not df.isnull().values.any():
        return df

    # Populate data frame if only half of it is complete
    for col in df:
        for lin in df[col].keys():
            df[lin][col] = df[col][lin]

    return df


def distance_of_route(route, table_of_distances):
    """
    Calculates the total distance traveled of a route

    :param route: Desired route
    :param table_of_distances: Table of distances containing each city pair
    :return: Total distance traveled
    """
    no_of_cities = len(route)
    total_distance = 0
    for city in range(no_of_cities):
        current_city = route[city]
        next_city = route[city + 1] if city < (no_of_cities - 1) else route[0]
        total_distance += int(table_of_distances[current_city][next_city])
    return total_distance
