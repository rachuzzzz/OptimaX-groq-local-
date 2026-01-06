import { Component, Input, OnInit, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Chart, ChartConfiguration, ChartType, registerables } from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';

Chart.register(...registerables, zoomPlugin);

export interface ChartData {
  type: ChartType;
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    backgroundColor?: string | string[];
    borderColor?: string | string[];
    borderWidth?: number;
  }[];
}

@Component({
  selector: 'app-chart-visualization',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './chart-visualization.html',
  styleUrl: './chart-visualization.scss'
})
export class ChartVisualizationComponent implements AfterViewInit {
  @Input() chartData!: ChartData;
  @Input() chartType: ChartType = 'bar';
  @Input() allowChartTypeSwitch: boolean = false; // Hide chart type buttons by default
  @ViewChild('chartCanvas', { static: false }) chartCanvas!: ElementRef<HTMLCanvasElement>;

  chart: Chart | null = null;
  currentType: ChartType = 'bar';
  showChart = true;

  ngAfterViewInit() {
    if (this.chartData && this.validateChartData()) {
      // Validate chart type before rendering
      const validTypes: ChartType[] = ['bar', 'line', 'pie', 'doughnut'];
      const requestedType = this.chartData.type || this.chartType;

      if (validTypes.includes(requestedType)) {
        this.currentType = requestedType;
      } else {
        console.warn(`Invalid chart type "${requestedType}". Falling back to "bar".`);
        this.currentType = 'bar';
        this.chartData.type = 'bar';
      }

      setTimeout(() => this.renderChart(), 100);
    } else {
      console.error('Invalid chart data provided:', this.chartData);
    }
  }

  /**
   * Validate that chart data is properly structured
   */
  validateChartData(): boolean {
    if (!this.chartData) {
      console.error('Chart data is undefined');
      return false;
    }
    if (!this.chartData.labels || this.chartData.labels.length === 0) {
      console.error('Chart data has no labels');
      return false;
    }
    if (!this.chartData.datasets || this.chartData.datasets.length === 0) {
      console.error('Chart data has no datasets');
      return false;
    }
    return true;
  }

  renderChart() {
    console.log(`[Chart Visualization] Rendering chart type: ${this.currentType}`);

    if (this.chart) {
      this.chart.destroy();
    }

    if (!this.validateChartData()) {
      console.error('[Chart Visualization] Cannot render chart: invalid data');
      return;
    }

    const ctx = this.chartCanvas.nativeElement.getContext('2d');
    if (!ctx) {
      console.error('[Chart Visualization] Failed to get canvas context');
      return;
    }

    console.log(`[Chart Visualization] Chart data:`, {
      type: this.currentType,
      labelCount: this.chartData.labels.length,
      datasetCount: this.chartData.datasets.length,
      labels: this.chartData.labels.slice(0, 5) // Log first 5 labels
    });

    const config: ChartConfiguration = {
      type: this.currentType,
      data: {
        labels: this.chartData.labels,
        datasets: this.chartData.datasets.map((dataset, index) => ({
          ...dataset,
          backgroundColor: dataset.backgroundColor || this.getColorGradient(ctx, index),
          borderColor: dataset.borderColor || this.getSolidColor(index),
          borderWidth: dataset.borderWidth || 2,
          // Enhanced visual properties
          tension: this.currentType === 'line' ? 0.4 : 0, // Smooth curves for line charts
          fill: this.currentType === 'line' ? true : false, // Fill area under line charts
          pointRadius: this.currentType === 'line' ? 4 : 0,
          pointHoverRadius: this.currentType === 'line' ? 6 : 0,
          pointBackgroundColor: dataset.borderColor || this.getSolidColor(index),
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          intersect: false,
          mode: 'index'
        },
        animation: {
          duration: 1200,
          easing: 'easeInOutQuart'
        },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: {
              color: '#e5e7eb',
              font: {
                size: 13,
                weight: 600,
                family: "'Inter', sans-serif"
              },
              padding: 20,
              usePointStyle: true,
              pointStyle: 'circle',
              boxWidth: 8,
              boxHeight: 8
            }
          },
          tooltip: {
            enabled: true,
            backgroundColor: 'rgba(17, 24, 39, 0.97)',
            titleColor: '#f9fafb',
            bodyColor: '#d1d5db',
            borderColor: 'rgba(59, 130, 246, 0.5)',
            borderWidth: 2,
            padding: 16,
            displayColors: true,
            boxPadding: 8,
            cornerRadius: 8,
            titleFont: {
              size: 14,
              weight: 'bold'
            },
            bodyFont: {
              size: 13
            },
            callbacks: {
              label: (context: any) => {
                let label = context.dataset.label || '';
                if (label) {
                  label += ': ';
                }
                if (context.parsed.y !== null) {
                  label += new Intl.NumberFormat('en-US', {
                    maximumFractionDigits: 2
                  }).format(context.parsed.y);
                }
                return label;
              }
            }
          },
          zoom: {
            zoom: {
              wheel: {
                enabled: true,
                speed: 0.1
              },
              pinch: {
                enabled: true
              },
              mode: 'x'
            },
            pan: {
              enabled: true,
              mode: 'x',
              modifierKey: 'ctrl'
            },
            limits: {
              x: {min: 'original', max: 'original'}
            }
          }
        },
        scales: this.currentType !== 'pie' && this.currentType !== 'doughnut' ? {
          y: {
            beginAtZero: true,
            grid: {
              color: 'rgba(107, 114, 128, 0.15)',
              lineWidth: 1
            },
            border: {
              display: false
            },
            ticks: {
              color: '#9ca3af',
              font: {
                size: 12,
                weight: 500
              },
              padding: 8,
              callback: function(value: any) {
                return new Intl.NumberFormat('en-US', {
                  notation: 'compact',
                  maximumFractionDigits: 1
                }).format(value);
              }
            }
          },
          x: {
            grid: {
              color: 'rgba(107, 114, 128, 0.08)',
              lineWidth: 1
            },
            border: {
              display: false
            },
            ticks: {
              color: '#9ca3af',
              font: {
                size: 12,
                weight: 500
              },
              padding: 8,
              maxRotation: 45,
              minRotation: 0
            }
          }
        } : undefined
      }
    };

    try {
      this.chart = new Chart(ctx, config);
      console.log(`[Chart Visualization] ✓ Chart rendered successfully`);
    } catch (error) {
      console.error('[Chart Visualization] Failed to create chart:', error);
      throw error;
    }
  }

  /**
   * Generate dynamic color gradients for chart datasets
   */
  getColorGradient(ctx: CanvasRenderingContext2D, index: number): CanvasGradient {
    const canvas = this.chartCanvas.nativeElement;
    const height = canvas.height || 400;
    const gradient = ctx.createLinearGradient(0, 0, 0, height);

    // Vibrant color palette matching the service
    const colorStops = [
      ['rgba(59, 130, 246, 0.8)', 'rgba(59, 130, 246, 0.2)'],    // Blue
      ['rgba(16, 185, 129, 0.8)', 'rgba(16, 185, 129, 0.2)'],    // Emerald
      ['rgba(249, 115, 22, 0.8)', 'rgba(249, 115, 22, 0.2)'],    // Orange
      ['rgba(139, 92, 246, 0.8)', 'rgba(139, 92, 246, 0.2)'],    // Purple
      ['rgba(236, 72, 153, 0.8)', 'rgba(236, 72, 153, 0.2)'],    // Pink
      ['rgba(245, 158, 11, 0.8)', 'rgba(245, 158, 11, 0.2)'],    // Amber
      ['rgba(20, 184, 166, 0.8)', 'rgba(20, 184, 166, 0.2)'],    // Teal
      ['rgba(239, 68, 68, 0.8)', 'rgba(239, 68, 68, 0.2)'],      // Red
    ];

    const selectedColor = colorStops[index % colorStops.length];
    gradient.addColorStop(0, selectedColor[0]);
    gradient.addColorStop(1, selectedColor[1]);

    return gradient;
  }

  /**
   * Get solid color for borders and points
   */
  getSolidColor(index: number): string {
    const solidColors = [
      'rgba(59, 130, 246, 1)',     // Blue
      'rgba(16, 185, 129, 1)',     // Emerald
      'rgba(249, 115, 22, 1)',     // Orange
      'rgba(139, 92, 246, 1)',     // Purple
      'rgba(236, 72, 153, 1)',     // Pink
      'rgba(245, 158, 11, 1)',     // Amber
      'rgba(20, 184, 166, 1)',     // Teal
      'rgba(239, 68, 68, 1)',      // Red
    ];

    return solidColors[index % solidColors.length];
  }

  changeChartType(type: ChartType) {
    if (this.currentType === type) {
      return; // Already displaying this type
    }

    console.log(`Switching chart type from ${this.currentType} to ${type}`);
    this.currentType = type;
    this.chartData.type = type;

    // Small delay for smooth transition
    setTimeout(() => {
      this.renderChart();
    }, 50);
  }

  downloadChart() {
    if (this.chart) {
      const link = document.createElement('a');
      link.download = `chart-${Date.now()}.png`;
      link.href = this.chart.toBase64Image();
      link.click();
    }
  }

  toggleView() {
    this.showChart = !this.showChart;
  }

  resetZoom() {
    if (this.chart) {
      this.chart.resetZoom();
    }
  }

  zoomIn() {
    if (this.chart) {
      this.chart.zoom(1.1);
    }
  }

  zoomOut() {
    if (this.chart) {
      this.chart.zoom(0.9);
    }
  }
}
