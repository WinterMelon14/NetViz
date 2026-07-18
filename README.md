# NetViz

NetViz is a local Windows application for visualizing how tensors move and
change during the forward pass of a PyTorch model. It inspects your model,
runs a representative forward pass, and presents the resulting operation graph
with tensor shapes, types, values, and diagnostics.

Your model and trace data stay on your computer. NetViz does not require a
system Python installation or a Vite development server.

## Download

NetViz 1.0.0 is distributed as:

`NetViz-windows-x64-1.0.0.zip`

Download the ZIP from the releases page.

NetViz 1.0.0 supports 64-bit Windows 10 and Windows 11. Microsoft Edge WebView2
Runtime is required.

## Install and run

NetViz does not use an installer.

1. Save `NetViz-windows-x64-1.0.0.zip` to your computer.
2. Right-click the ZIP and select **Extract All**.
3. Open the extracted folder.
4. Run `NetViz.exe`.

Keep `NetViz.exe` and the `_internal` folder together. Do not run the executable
from inside the ZIP, and do not copy `NetViz.exe` by itself.

NetViz 1.0.0 is unsigned, so Windows SmartScreen may display a warning. If you
received the ZIP from a source you trust, select **More info**, verify that the
application is NetViz, and then select **Run anyway**.

If Windows reports that WebView2 is unavailable, install the Microsoft Edge
WebView2 Runtime from Microsoft and start NetViz again.

## Trace a model

1. Start NetViz and select **Trace Model**.
2. Choose a Python file, or switch to the paste option and paste Python source.
3. Wait for NetViz to inspect the source, then select the model class to trace.
4. Enter any required constructor arguments. Constructor values are entered as
   JSON literals, such as `32`, `true`, `[64, 128]`, or `"relu"`.
5. Configure representative inputs for the model's `forward` parameters. For
   tensor inputs, choose the shape, data type, and value generator shown in the
   editor. If the model defines `netviz_example_inputs()`, you can choose to use
   those inputs instead.
6. Review the compatibility report and project scope. Blocking findings must be
   resolved before NetViz can run the model.
7. Optionally enable CPU profiling and choose the warmup and measurement counts.
8. Read the trusted-code notice and confirm that you trust the exact inspected
   source.
9. Select **Run Trace**.

Tracing can be cancelled while it is running. A run that exceeds the configured
timeout is terminated and reported as a timeout rather than leaving its worker
running indefinitely.

## Explore a trace

After a successful trace:

- Select a graph node to inspect its operation and observed tensor information.
- Follow edges to see how values flow between operations.
- Use **Fit graph** in the toolbar to bring the full graph back into view.
- Open **Settings** to adjust graph layout and display options.
- Select **Trace Model** again to trace another model or input configuration.

CPU profiling is optional. When enabled, NetViz performs additional forward
passes and reports timing aggregates; warmup runs are excluded from those
aggregates.

## Supported model environment

The NetViz executable contains its own Python runtime and its own copies of
PyTorch and NumPy. It does not use packages installed in your system Python,
Conda environment, or virtual environment.

NetViz 1.0.0 supports model code that imports:

- Python standard-library modules.
- The bundled `torch` package.
- The bundled `numpy` package.
- Local Python modules and resources discovered in the selected model's project
  directory.

Packages that are not bundled, such as `torchvision` and `transformers`, are not
available in the frozen 1.0.0 runtime. Installing them into another Python
environment will not make them available to NetViz.

Local imports and relative model resources should be kept alongside the selected
model file. NetViz displays the discovered project scope before execution and
will ask you to inspect again if a model, local module, or declared resource
changes after inspection.

## Security and privacy

NetViz runs locally, but tracing imports and executes the selected Python code
with your normal Windows user permissions. The worker process provides lifecycle
and crash isolation; it is not a security sandbox.

Only trace code that you wrote or obtained from a source you trust. Model code
can read, modify, or delete anything that your Windows account is permitted to
access.

## Troubleshooting

### NetViz does not start

- Make sure the ZIP was fully extracted.
- Confirm that `NetViz.exe` is still next to `_internal`.
- Install or repair Microsoft Edge WebView2 Runtime.
- Move the extracted folder to a normal local directory if Windows or a sync
  provider is blocking files in the current location.

### Windows blocks the executable

NetViz 1.0.0 is unsigned. Only if you trust where the ZIP came from, use
**More info** and **Run anyway** in the SmartScreen dialog.

### A model reports an unavailable import

The executable cannot use packages installed in your system Python environment.
Use the bundled `torch` and `numpy`, keep required local modules with the model,
or modify the model to remove unsupported third-party imports.

### The source changed or must be inspected again

NetViz checks the inspected model and project files before running them. Return
to the source step and inspect the current version again.

### A trace fails because of its inputs

Check that the representative tensor shapes and data types match the model's
`forward` method. When NetViz offers an input suggestion, review it before using
**Apply Suggestion** and running the trace again.

## Uninstall

Close NetViz and delete its extracted folder. NetViz does not install a separate
Python runtime, npm packages, or a background service on the user's computer.

## License

NetViz is proprietary software. See [LICENSE](LICENSE) for the applicable terms.
