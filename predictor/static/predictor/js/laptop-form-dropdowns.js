/**
 * Shared dropdown logic for LaptopSpecsForm: brand→series, cpu_brand→cpu_family/suffix/generation.
 * Call initLaptopFormDropdowns(prefix) on DOMContentLoaded.
 * For single form use prefix '' (empty string). For compare use 'lap1', 'lap2', 'lap3'.
 */
(function(global) {
    var brandSeriesMap = {
        'LENOVO': ['IDEAPAD', 'THINKPAD', 'LEGION', 'YOGA', 'UNKNOWN'],
        'HP': ['PAVILION', 'OMEN', 'ELITEBOOK', 'PROBOOK', 'SPECTRE', 'ENVY', 'VICTUS', 'UNKNOWN'],
        'DELL': ['INSPIRON', 'LATITUDE', 'XPS', 'ALIENWARE', 'VOSTRO', 'PRECISION', 'G SERIES', 'UNKNOWN'],
        'ASUS': ['VIVOBOOK', 'ZENBOOK', 'ROG', 'TUF', 'EXPERTBOOK', 'UNKNOWN'],
        'ACER': ['ASPIRE', 'NITRO', 'PREDATOR', 'SWIFT', 'SPIN', 'UNKNOWN'],
        'APPLE': ['MACBOOK AIR', 'MACBOOK PRO'],
        'MSI': ['STEALTH', 'KATANA', 'RAIDER', 'PRESTIGE', 'MODERN', 'UNKNOWN'],
        'GIGABYTE': ['AERO', 'AORUS', 'G5', 'UNKNOWN'],
        'HUAWEI': ['MATEBOOK', 'UNKNOWN'],
        'SAMSUNG': ['GALAXY BOOK', 'UNKNOWN'],
        'TOSHIBA': ['SATELLITE', 'TECRA', 'UNKNOWN'],
        'OTHER': ['UNKNOWN']
    };
    var seriesDisplayNames = {
        'IDEAPAD': 'IdeaPad', 'THINKPAD': 'ThinkPad', 'LEGION': 'Legion', 'YOGA': 'Yoga',
        'PAVILION': 'Pavilion', 'OMEN': 'Omen', 'VICTUS': 'Victus', 'ELITEBOOK': 'EliteBook',
        'PROBOOK': 'ProBook', 'SPECTRE': 'Spectre', 'ENVY': 'Envy', 'INSPIRON': 'Inspiron',
        'LATITUDE': 'Latitude', 'XPS': 'XPS', 'ALIENWARE': 'Alienware', 'VOSTRO': 'Vostro',
        'PRECISION': 'Precision', 'G SERIES': 'G Series', 'VIVOBOOK': 'VivoBook', 'ZENBOOK': 'ZenBook',
        'ROG': 'ROG', 'TUF': 'TUF', 'EXPERTBOOK': 'ExpertBook', 'ASPIRE': 'Aspire', 'NITRO': 'Nitro',
        'PREDATOR': 'Predator', 'SWIFT': 'Swift', 'SPIN': 'Spin', 'MACBOOK AIR': 'MacBook Air',
        'MACBOOK PRO': 'MacBook Pro', 'AERO': 'Aero', 'AORUS': 'Aorus', 'G5': 'G5',
        'STEALTH': 'Stealth', 'KATANA': 'Katana', 'RAIDER': 'Raider', 'PRESTIGE': 'Prestige',
        'MODERN': 'Modern', 'MATEBOOK': 'MateBook', 'GALAXY BOOK': 'Galaxy Book',
        'SATELLITE': 'Satellite', 'TECRA': 'Tecra', 'UNKNOWN': 'Unknown/Other'
    };
    var cpuBrandFamilyMap = {
        'INTEL': [
            {value: 'i3', label: 'Intel Core i3'}, {value: 'i5', label: 'Intel Core i5'},
            {value: 'i7', label: 'Intel Core i7'}, {value: 'i9', label: 'Intel Core i9'},
            {value: 'ultra5', label: 'Intel Core Ultra 5'}, {value: 'ultra7', label: 'Intel Core Ultra 7'},
            {value: 'ultra9', label: 'Intel Core Ultra 9'}, {value: 'celeron', label: 'Intel Celeron'},
            {value: 'pentium', label: 'Intel Pentium'}, {value: 'xeon', label: 'Intel Xeon'}
        ],
        'AMD': [
            {value: 'r3', label: 'AMD Ryzen 3'}, {value: 'r5', label: 'AMD Ryzen 5'},
            {value: 'r7', label: 'AMD Ryzen 7'}, {value: 'r9', label: 'AMD Ryzen 9'}
        ],
        'APPLE': [
            {value: 'm1', label: 'Apple M1'}, {value: 'm2', label: 'Apple M2'},
            {value: 'm3', label: 'Apple M3'}, {value: 'm4', label: 'Apple M4'}
        ],
        'QUALCOMM': [{value: 'SNAPDRAGON', label: 'Qualcomm Snapdragon'}],
        'UNKNOWN': [{value: 'UNKNOWN', label: 'Unknown'}]
    };
    var noGenerationChips = ['m1', 'm2', 'm3', 'm4', 'SNAPDRAGON', 'UNKNOWN'];
    var cpuSuffixMap = {
        'INTEL': [
            {value: '', label: 'None/Standard'}, {value: 'U', label: 'U (Ultra-low power)'},
            {value: 'H', label: 'H (High performance)'}, {value: 'HQ', label: 'HQ (High performance Quad)'},
            {value: 'HS', label: 'HS (High performance Slim)'}, {value: 'HX', label: 'HX (Extreme performance)'},
            {value: 'G', label: 'G (With integrated graphics)'}, {value: 'P', label: 'P (Performance)'}
        ],
        'AMD': [
            {value: '', label: 'None/Standard'}, {value: 'U', label: 'U (Ultra-low power)'},
            {value: 'H', label: 'H (High performance)'}, {value: 'HS', label: 'HS (High performance Slim)'},
            {value: 'HX', label: 'HX (Extreme performance)'}
        ],
        'APPLE': [
            {value: '', label: 'None/Standard'}, {value: 'PRO', label: 'Pro'}, {value: 'MAX', label: 'Max'}
        ],
        'QUALCOMM': [{value: '', label: 'None/Standard'}],
        'UNKNOWN': [{value: '', label: 'None/Standard'}]
    };
    var ultraFamilies = ['ultra5', 'ultra7', 'ultra9'];
    var ryzenFamilies = ['r3', 'r5', 'r7', 'r9'];

    function initLaptopFormDropdowns(prefix) {
        var id = function(name) {
            if (prefix)
                return document.getElementById('id_' + prefix + '-' + name);
            if (name === 'generation-container' || name === 'generation-help' || name === 'ultra-note')
                return document.getElementById(name);
            return document.getElementById('id_' + name);
        };

        function updateSeriesDropdown() {
            var brandSelect = id('brand');
            var seriesSelect = id('series');
            if (!brandSelect || !seriesSelect) return;
            var selectedBrand = brandSelect.value;
            var currentSeries = seriesSelect.value;
            seriesSelect.innerHTML = '<option value="">Select Series</option>';
            if (selectedBrand && brandSeriesMap[selectedBrand]) {
                brandSeriesMap[selectedBrand].forEach(function(series) {
                    var option = document.createElement('option');
                    option.value = series;
                    option.textContent = seriesDisplayNames[series] || series;
                    seriesSelect.appendChild(option);
                });
                if (brandSeriesMap[selectedBrand].indexOf(currentSeries) !== -1) {
                    seriesSelect.value = currentSeries;
                }
            }
            if (selectedBrand === 'APPLE') {
                var cb = id('cpu_brand');
                if (cb) { cb.value = 'APPLE'; updateCpuFamilyDropdown(); }
            }
        }

        function updateCpuFamilyDropdown() {
            var cpuBrandSelect = id('cpu_brand');
            var cpuFamilySelect = id('cpu_family');
            if (!cpuBrandSelect || !cpuFamilySelect) return;
            var selectedCpuBrand = cpuBrandSelect.value;
            var currentFamily = cpuFamilySelect.value;
            cpuFamilySelect.innerHTML = '<option value="">Select CPU Family</option>';
            if (selectedCpuBrand && cpuBrandFamilyMap[selectedCpuBrand]) {
                cpuBrandFamilyMap[selectedCpuBrand].forEach(function(cpu) {
                    var option = document.createElement('option');
                    option.value = cpu.value;
                    option.textContent = cpu.label;
                    cpuFamilySelect.appendChild(option);
                });
                var validValues = cpuBrandFamilyMap[selectedCpuBrand].map(function(c) { return c.value; });
                if (validValues.indexOf(currentFamily) !== -1) cpuFamilySelect.value = currentFamily;
            }
            updateGenerationVisibility();
            updateCpuSuffixDropdown();
        }

        function updateCpuSuffixDropdown() {
            var cpuBrandSelect = id('cpu_brand');
            var cpuSuffixSelect = id('cpu_suffix');
            if (!cpuBrandSelect || !cpuSuffixSelect) return;
            var selectedCpuBrand = cpuBrandSelect.value;
            var currentSuffix = cpuSuffixSelect.value;
            cpuSuffixSelect.innerHTML = '';
            var suffixes = cpuSuffixMap[selectedCpuBrand] || cpuSuffixMap['INTEL'];
            suffixes.forEach(function(suffix) {
                var option = document.createElement('option');
                option.value = suffix.value;
                option.textContent = suffix.label;
                cpuSuffixSelect.appendChild(option);
            });
            var validValues = suffixes.map(function(s) { return s.value; });
            if (validValues.indexOf(currentSuffix) !== -1) cpuSuffixSelect.value = currentSuffix;
        }

        function updateGenerationVisibility() {
            var cpuFamilySelect = id('cpu_family');
            var generationSelect = id('cpu_generation');
            if (!cpuFamilySelect || !generationSelect) return;
            var selectedFamily = cpuFamilySelect.value;
            var generationContainer = id('generation-container');
            var generationHelp = id('generation-help');
            if (noGenerationChips.indexOf(selectedFamily) !== -1) {
                generationSelect.value = '0';
                generationSelect.disabled = true;
                if (generationHelp) generationHelp.textContent = 'N/A for this CPU type';
            } else {
                generationSelect.disabled = false;
                if (generationHelp) generationHelp.textContent = '';
            }
            handleCpuGeneration();
        }

        function handleCpuGeneration() {
            var cpuFamilySelect = id('cpu_family');
            var cpuGenerationSelect = id('cpu_generation');
            if (!cpuFamilySelect || !cpuGenerationSelect) return;
            var selectedFamily = cpuFamilySelect.value;
            var ultraNote = id('ultra-note');
            if (ultraNote) ultraNote.style.display = 'none';

            if (ultraFamilies.indexOf(selectedFamily) !== -1) {
                if (ultraNote) ultraNote.style.display = '';
                for (var i = 0; i < cpuGenerationSelect.options.length; i++) {
                    var opt = cpuGenerationSelect.options[i];
                    opt.style.display = ['1', '2', '3', '0'].indexOf(opt.value) !== -1 ? '' : 'none';
                }
                if (['1', '2', '3', '0'].indexOf(cpuGenerationSelect.value) === -1) cpuGenerationSelect.value = '1';
            } else if (ryzenFamilies.indexOf(selectedFamily) !== -1) {
                for (var i = 0; i < cpuGenerationSelect.options.length; i++) {
                    var opt = cpuGenerationSelect.options[i];
                    var genValue = parseInt(opt.value, 10);
                    opt.style.display = (opt.value === '0' || (genValue >= 1 && genValue <= 10)) ? '' : 'none';
                }
                var currentGen = parseInt(cpuGenerationSelect.value, 10);
                if (currentGen > 10) cpuGenerationSelect.value = '7';
            } else {
                for (var i = 0; i < cpuGenerationSelect.options.length; i++) {
                    cpuGenerationSelect.options[i].style.display = '';
                }
            }
        }

        var brandSelect = id('brand');
        var cpuBrandSelect = id('cpu_brand');
        var cpuFamilySelect = id('cpu_family');
        if (brandSelect) brandSelect.addEventListener('change', updateSeriesDropdown);
        if (cpuBrandSelect) cpuBrandSelect.addEventListener('change', updateCpuFamilyDropdown);
        if (cpuFamilySelect) {
            cpuFamilySelect.addEventListener('change', function() {
                updateGenerationVisibility();
                handleCpuGeneration();
            });
        }
        updateSeriesDropdown();
        updateCpuFamilyDropdown();
        updateCpuSuffixDropdown();
        if (cpuFamilySelect && cpuFamilySelect.value) {
            updateGenerationVisibility();
            handleCpuGeneration();
        }
    }

    global.initLaptopFormDropdowns = initLaptopFormDropdowns;
})(typeof window !== 'undefined' ? window : this);
