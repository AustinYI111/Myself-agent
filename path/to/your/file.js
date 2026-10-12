function equityChartOptions() {
    return {
        type: 'line',
        data: {}, // existing data options
        options: {
            interactions: {
                mode: 'index', // changed interaction mode to index
                intersect: false // prevent the hover effect from directly intersecting data points
            },
            onHover: function (event, chartElement) {
                if (chartElement.length) {
                    event.target.style.cursor = 'pointer'; // Change cursor on hover
                } else {
                    event.target.style.cursor = 'default';
                }
            }
        }
    };
}