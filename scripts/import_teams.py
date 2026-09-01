from app.importers.team_importer import TeamImporter


def main():
    success = TeamImporter().run()

    if success:
        print("Импорт команд завершён.")
    else:
        print("Импорт команд завершился с ошибкой.")


if __name__ == "__main__":
    main()
