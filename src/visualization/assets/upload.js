/**
 * upload.js
 *
 * Script called by the "uploaded landing page" of the visualization/interface to tell the
 * API to process a name/CSV file into the database.
 */
(() => {
    // Detect if user used a name rather than a CSV file
    const isSingle = document.querySelector('meta[name="plural False"]')
    if (isSingle) {
        // Get the company name and send it to the API
        const companyName = document.getElementById('company-name').textContent
        fetch(`/api/add_company?name=${encodeURIComponent(companyName)}`)
            .then(() => {
                window.close()
            })
    } else {
        // Get the CSV file path and send it to the API
        const pathName = document.querySelector('meta[name="path"]').dataset.value
        fetch(`/api/add_company?csv=${encodeURIComponent(pathName)}`)
            .then(() => {
                window.close()
            })
    }
})()
