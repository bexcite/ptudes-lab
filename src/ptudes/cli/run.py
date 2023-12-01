import click

from ptudes.cli.flyby import ptudes_flyby
from ptudes.cli.viz import ptudes_viz

@click.group(name="ptudes")
def ptudes_cli() -> None:
    """Ptudes (point tudes) - viz, slam and mapping playground.

    Various experiments with mainly Ouster Lidar and other sensors.
    """
    pass

ptudes_cli.add_command(ptudes_flyby)
ptudes_cli.add_command(ptudes_viz)

def main():
    ptudes_cli()

if __name__ == '__main__':
     main()
