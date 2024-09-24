import click

try:
    from ouster.sdk import client
except ImportError:
    print(
        f'''The ouster-sdk package is not installed on your system. The required version
is 0.11.0 or higher.

Please install it manually by running two commands one by one, disregard any errors:

pip3 install 'ouster-sdk>=0.11.0'
pip3 install 'kiss-icp>=1.0.0'

Check versions of kiss-icp and ouster-sdk installed and make sure they are in the
range of above commands:

pip3 list

NOTE: We have not included ouster-sdk in the project dependencies because it
      requires kiss-icp==0.4.0, which is not compatible with this project.
      This project supports the latest version of kiss-icp (1.0.0 or higher)
      and ouster-sdk>=0.11.0.
''')
    exit(1)

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
