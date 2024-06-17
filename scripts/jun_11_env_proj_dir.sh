RUNDIR="$PROJDIR/runs"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

echo $OUTDIR