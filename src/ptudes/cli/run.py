import click

from ptudes.cli.flyby import ptudes_flyby
from ptudes.cli.viz import ptudes_viz
from ptudes.cli.stat import ptudes_stat
# from ptudes.cli.odom import ptudes_odom
from ptudes.cli.ekf_bench import ptudes_ekf_bench

@click.group(name="ptudes")
def ptudes_cli() -> None:
    """P(oint) (e)Tudes - viz, slam and mapping playground.

    Various experiments with mainly Ouster Lidar and other sensors.
    """
    pass

ptudes_cli.add_command(ptudes_flyby)
ptudes_cli.add_command(ptudes_viz)
# ptudes_cli.add_command(ptudes_odom)
ptudes_cli.add_command(ptudes_stat)

ptudes_cli.add_command(ptudes_ekf_bench)

def main():
    ptudes_cli()

if __name__ == '__main__':
     main()
