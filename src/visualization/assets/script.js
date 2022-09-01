/**
 * script.js
 *
 * Script called by the main dashboard of the visualization/interface to handle user input
 * and populate the dashboard based on data from the database via the API.
 */
const selCompany = document.getElementById('sel-company')
const linkContainer = document.getElementById('link-container')
const linkBox = document.getElementById('links')
const linkA = document.getElementById('link-a')
const sentenceContainer = document.getElementById('sentence-container')
const inspectionContainer = document.getElementById('link-inspection-container')
const entityBox = document.getElementById('top-entity-box')
let apiData = null;

// Called on page load. Fetch company info from the API and populate the company list.
(async function () {
    const cmpContainer = document.getElementById('company-list')
    while (cmpContainer.firstChild) {
        cmpContainer.removeChild(cmpContainer.firstChild)
    }
    const nameData = await (await fetch('/api/companies')).json()
    for (const company of nameData.companies) {
        const el = document.createElement('div')
        el.classList.add('company-list-name')
        el.appendChild(document.createElement('span'))
        el.dataset.confidence = company.confidenceScore
        el.firstChild.textContent = company.name
        el.addEventListener('click', () => {
            const selected = document.querySelector('.company-list-name.selected')
            if (selected) selected.classList.remove('selected')
            el.classList.add('selected')
            update(company.name)
            return false
        })
        cmpContainer.appendChild(el)
    }
    update(nameData.companies[0].name)
})()

// Called when a company name is clicked. Load URL data for a company from the API into the second list.
async function update(name) {
    linkContainer.classList.add('loading')
    inspectionContainer.classList.add('hidden')
    selCompany.textContent = name
    apiData = await (await fetch('/api/fetch_data/' + name)).json()
    clearInspectBox()
    while (linkBox.firstChild) {
        linkBox.removeChild(linkBox.firstChild)
    }

    let latestDate = 0
    let latestElements = []
    for (const point of apiData.data) {
        const link = document.createElement('div')
        const timestamp = new Date(point.dateProcessed).getTime()
        if (timestamp > latestDate) {
            latestElements = [link]
            latestDate = timestamp
        } else if (timestamp === latestDate) {
            latestElements.push(link)
        }
        link.classList.add('link-box')
        link.appendChild(document.createElement('span'))
        link.firstChild.textContent = point.url
        link.dataset.confidence = point.confidenceScore
        link.addEventListener('click', () => {
            inspectURL(link, point.url)
        })
        linkBox.appendChild(link)
    }
    for (const el of latestElements) {
        el.classList.add('recent')
    }
    linkA.href = ''
    linkA.firstChild.textContent = ''
    linkContainer.classList.remove('loading')
}

// Called when a URL from the second list is clicked. Display sentence-level data within the third pane.
function inspectURL(el, url) {
    linkA.href = url
    const selected = document.querySelector('.link-box.selected')
    if (selected) selected.classList.remove('selected')
    el.classList.add('selected')
    linkA.firstChild.textContent = url

    clearInspectBox()

    const point = apiData.data.find(l => l.url === url)
    const allCompanies = {}

    for (const sentence of point.sentences.sort((a, b) => a.length - b.length)) {
        const sentenceElement = document.createElement('div')
        sentenceElement.classList.add('sentence-box')
        sentenceElement.classList.add('sentence-' + sentence.type)
        sentenceElement.dataset.confidence = sentence.confidenceScore
        sentenceElement.appendChild(document.createElement('span'))
        sentenceElement.firstChild.innerText = sentence.text
        for (const company of sentence.companies) {
            sentenceElement.firstChild.innerHTML = sentenceElement.firstChild.innerHTML.replaceAll(company, `<span class="entity-label">${company}</span>`)
            if (!allCompanies[company]) {
                allCompanies[company] = (sentenceElement.firstChild.innerHTML.match(new RegExp(company, 'g')) || []).length
            } else {
                allCompanies[company] += (sentenceElement.firstChild.innerHTML.match(new RegExp(company, 'g')) || []).length
            }
        }

        sentenceContainer.appendChild(sentenceElement)
    }
    if (Object.keys(allCompanies).length > 1) {
        const sortedEntities = Object.keys(allCompanies).sort((a, b) => allCompanies[b] - allCompanies[a]).filter(a => allCompanies[a] > 0)
        entityBox.innerHTML = sortedEntities.map(e => `<span class="entity-label">${e} (${allCompanies[e]})</span>`).join(', ')
    } else {
        entityBox.innerHTML = '<span>None detected</span>'
    }
    inspectionContainer.classList.remove('hidden')
}

// Called when a user wants to add a company/csv file to the system for processing.
function addCompanyPrompt(name) {
    const interimString = name.endsWith('.csv') ? `apply ${name} to the database` : `add a new entry for ${name}`
    const confirmed = confirm(`Are you sure you want to ${interimString}?\n` + 'This action will perform a web search and compare results against the program\'s machine learning model.')
    if (!confirmed) document.getElementById('csvinput').value = ''
    return confirmed
}

function clearInspectBox() {
    while (sentenceContainer.firstChild) {
        sentenceContainer.removeChild(sentenceContainer.firstChild)
    }
}
