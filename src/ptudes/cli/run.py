import click

from ptudes.cli.flyby import ptudes_flyby
from ptudes.cli.viz import ptudes_viz
from ptudes.cli.odom import ptudes_odom

@click.group(name="ptudes")
def ptudes_cli() -> None:
    """P(oint) (e)Tudes - viz, slam and mapping playground.

    Various experiments with mainly Ouster Lidar and other sensors.
    """
    pass

ptudes_cli.add_command(ptudes_flyby)
ptudes_cli.add_command(ptudes_viz)
ptudes_cli.add_command(ptudes_odom)

def main():
    ptudes_cli()

if __name__ == '__main__':
     main()
