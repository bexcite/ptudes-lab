import click
from ptudes.bag import print_bag_info

@click.group(name="bag")
def ptudes_bag():
    """Bag related helper commands."""
    pass

@click.command(name="info")
@click.argument('filename')
def info(filename):
    """Print ROS bag info by filename."""
    print_bag_info(filename)

ptudes_bag.add_command(info)