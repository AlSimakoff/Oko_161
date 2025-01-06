import settings

import db
from db import add_entry


def main(database):
    db.initiate(database)



if __name__ == "__main__":
    main(
        settings.database_path
    )