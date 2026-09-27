from dubsync.runtime import configure_nvidia_dll_paths

configure_nvidia_dll_paths()

from dubsync.cli import main


if __name__ == "__main__":
    main()