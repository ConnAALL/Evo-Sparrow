for i in {1..10}
do
    echo "Running model_run.py iteration $i..."
    python3 model_run.py
done

echo "All runs completed."