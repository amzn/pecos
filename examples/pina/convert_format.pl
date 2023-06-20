my $inpfile;
open($inpfile,"<",$ARGV[0]);

my $ftfile;
open($ftfile,">",$ARGV[1]);

my $lblfile;
open($lblfile,">",$ARGV[2]);

my $ctr = 0;
while(<$inpfile>)
{
	chomp($_);
	
	if($ctr==0)
	{
		my @items = split(" ",$_);
		$num_inst = $items[0];
		$num_ft = $items[1];
		$num_lbl = $items[2];

		print $ftfile "$num_inst $num_ft\n";
		print $lblfile "$num_inst $num_lbl\n";
	}
	else
	{
		my @items = split(" ",$_,2);
		
		if($_ =~ /^ .*/)
		{
			print $lblfile "\n";
			print $ftfile $items[0]."\n"; 
		}
		else
		{
			my @lbls = split(",",$items[0]);
			print $lblfile join(" ",map {"$_:1"} @lbls)."\n";
			print $ftfile $items[1]."\n"; 
		}
	}

	$ctr++;
}

close($inpfile);
close($ftfile);
close($lblfile);
