import { Component, Input, OnInit, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Chart, ChartConfiguration, ChartType, registerables } from 'chart.js';

Chart.register(...registerables);

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
  @ViewChild('chartCanvas', { static: false }) chartCanvas!: ElementRef<HTMLCanvasElement>;

  chart: Chart | null = null;
  currentType: ChartType = 'bar';
  showChart = true;

  ngAfterViewInit() {
    if (this.chartData) {
      this.currentType = this.chartData.type || this.chartType;
      setTimeout(() => this.renderChart(), 100);
    }
  }

  renderChart() {
    if (this.chart) {
      this.chart.destroy();
    }

    const ctx = this.chartCanvas.nativeElement.getContext('2d');
    if (!ctx) return;

    const config: ChartConfiguration = {
      type: this.currentType,
      data: {
        labels: this.chartData.labels,
        datasets: this.chartData.datasets.map(dataset => ({
          ...dataset,
          backgroundColor: dataset.backgroundColor || this.getGoldGradient(ctx),
          borderColor: dataset.borderColor || 'rgba(255, 215, 0, 1)',
          borderWidth: dataset.borderWidth || 2
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 1000,
          easing: 'easeInOutQuart'
        },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: {
              color: '#e5e7eb',
              font: {
                size: 12,
                weight: 500
              },
              padding: 15
            }
          },
          tooltip: {
            backgroundColor: 'rgba(30, 30, 35, 0.95)',
            titleColor: '#ffd700',
            bodyColor: '#e5e7eb',
            borderColor: 'rgba(212, 175, 55, 0.5)',
            borderWidth: 1,
            padding: 12,
            displayColors: true,
            boxPadding: 6
          }
        },
        scales: this.currentType !== 'pie' && this.currentType !== 'doughnut' ? {
          y: {
            beginAtZero: true,
            grid: {
              color: 'rgba(255, 215, 0, 0.1)'
            },
            ticks: {
              color: '#9ca3af',
              font: {
                size: 11
              }
            }
          },
          x: {
            grid: {
              color: 'rgba(255, 215, 0, 0.05)'
            },
            ticks: {
              color: '#9ca3af',
              font: {
                size: 11
              }
            }
          }
        } : undefined
      }
    };

    this.chart = new Chart(ctx, config);
  }

  getGoldGradient(ctx: CanvasRenderingContext2D): CanvasGradient {
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(255, 215, 0, 0.8)');
    gradient.addColorStop(1, 'rgba(212, 175, 55, 0.6)');
    return gradient;
  }

  changeChartType(type: ChartType) {
    this.currentType = type;
    this.chartData.type = type;
    this.renderChart();
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
}
