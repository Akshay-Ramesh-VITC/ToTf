1. Training Monitor

# Simulation of a training loop
epochs = 5
steps_per_epoch = 100

with TrainingMonitor(total_steps=epochs * steps_per_epoch) as monitor:
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            # 1. Your training logic here...
            time.sleep(0.05) 
            
            # 2. Mock metrics
            current_loss = 0.5 / (step + 1)
            current_acc = 0.8 + (step / 1000)
            
            # 3. Update the monitor
            global_step = (epoch * steps_per_epoch) + step
            monitor.update(
                step=global_step, 
                metrics={"loss": current_loss, "acc": current_acc},
                prefix=f"Epoch {epoch+1}/{epochs}"
            )