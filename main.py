import settings

import db
from db import add_entry


def main(database):
    db.initiate(database)
    #add_entry(database,"Journal", {'time':'06.01.2025:12.39', 'color' : 'red', 'license_number' : 'A421TH61','type_auto':'truck'})
    res=db.select_data(database,"Journal")
    print(res)


if __name__ == "__main__":
    main(
        settings.database_path
    )