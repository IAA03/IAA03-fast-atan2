# IAA03_fast_math

## High-Performance Branchless Trigonometry for C++11.

IAA03_fast_math is a single-header math kernel designed to eliminate the "Trigonometry Tax" in high-throughput systems (Physics Engines, Audio DSP, and ML Pre-processing). By utilizing Branchless Octant Folding , ILP and SIMD (AVX2/SSE4.1), it achieves up to a 178x per-element throughput speedup over std::atan2.

*benchmarked on :11th Gen Intel® Core™ i5-1155G7 processor 4 cores, Max Turbo Frequency 4.5 GHz using the accompenying tests header*
<table style="border: 2px solid blue;border-radius: 2px">
  <tr>
    <th>Implementation</th>
    <th>Time(max/min) (1B calls)</th>
    <th>Per Element</th>
    <th>Variance(max / min)</th>
    <th>execution time Speedup</th>
    <th>thoughput speedup</th>
  </tr>
  <tr>
    <td>std::atan2</td>
    <td>28.49 ms</td>
    <td>28.49 ns</td>
    <td>(34.31 / 24.19) ms</td>
    <td>1.0x</td>
    <td>1.0x</td>
  </tr>
  <tr>
    <th colspan="2"><bold>IAA03 High Accuracy Atan2</bold></th>
  </tr>
  <tr>
    <td>Scalar</td>
    <td>16.6105 ms</td>
    <td>16.6105 ns</td>
    <td>(24.8781 / 13.9479) ms</td>
    <td>1.71x</td>
    <td>1.71x</td>
  </tr>
  <tr>
    <td>SIMD4 (SSE)</td>
    <td>3.02508 ms</td>
    <td>0.75627 ns</td>
    <td>(4.44005 / 2.07275)ms</td>
    <td>9.4x</td>
    <td>37.68x</td>
  </tr>
  <tr>
    <td>SIMD8 (AVX2)</td>
    <td>1.67714 ms</td>
    <td>0.209642 ns</td>
    <td>(2.53186 / 1.25685) ms</td>
    <td>17.0x</td>
    <td>136.32x</td>
  </tr>
  <tr>
    <th colspan="2"><bold>IAA03 Fast Atan2</bold></th>
  </tr>
  <tr>
    <td>Scalar</td>
    <td>11.2278 ms</td>
    <td>11.2278 ns</td>
    <td>(15.9596 / 8.99204) ms</td>
    <td>2.5x</td>
    <td>2.54x</td>
  </tr>
  <tr>
    <td>SIMD4 (SSE)</td>
    <td>2.08378 ms</td>
    <td>0.520946 ns</td>
    <td>(3.04052 / 1.35319)ms</td>
    <td>13.0x</td>
    <td>54.68x</td>
  </tr>
  <tr>
    <td>SIMD8 (AVX2)</td>
    <td>1.27845 ms</td>
    <td>0.159806 ns</td>
    <td>(1.79751 / 1.03329) ms</td>
    <td>22.3x</td>
    <td>178.06x</td>
  </tr>
</table>

<table>
  <tr>
    <th>Type</th>
    <th>Max error radians(degrees)</th>
    <th>Avrg error radians(degrees)</th>
    <th>Absolute error & Symmetry Verification</th>
  </tr>
  <tr>
    <th>High Accuracy</th>
    <th>2.57492e-05 radians (1.47532e-3 degrees)</th>
    <th>1.05749e-05 radians (6.059e-4 degrees)</th>
    <td><details>
  <summary>📊 Click to view detailed Error Distribution Plots</summary>
  <br>
  <img src="assets/accurate_360plot.png" width="600">
  <p><i>Note: Observe how the error collapses to zero at the 0, 90, 180, and 270 degree axes.</i></p>
</details></td>
  </tr>
  <tr>
    <th>Fast</th>
    <th>4.37665e-3 radians (2.50764e-1 degrees)</th>
    <th>2.09574e-3 radians (1.20077e-1 degrees)</th>
    <td><details>
  <summary>📊 Click to view detailed Error Distribution Plots</summary>
  <br>
  <img src="assets/fast_360plot.png" width="600">
  <p><i>Note: Observe how the error collapses to zero at the 0, 90, 180, and 270 degree axes.</i></p>
</details></td></td>
  </tr>
</table>

