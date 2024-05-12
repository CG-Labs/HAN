import re

def read_cv_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def parse_personal_details(cv_text):
    # Use regular expressions to find personal details
    name_match = re.search(r'CURRICULUM VITAE\s+(.+)', cv_text)
    contact_match = re.search(r'(\d{3} \d{8})\s+(\S+@\S+)', cv_text)

    personal_details = {
        'name': name_match.group(1).strip() if name_match else 'N/A',
        'phone': contact_match.group(1).strip() if contact_match else 'N/A',
        'email': contact_match.group(2).strip() if contact_match else 'N/A'
    }
    return personal_details

def parse_key_skills(cv_text):
    # Find the KEY SKILLS section and extract the skills
    skills_match = re.search(r'KEY SKILLS\s+(.+?)\n\n', cv_text, re.DOTALL)
    skills = skills_match.group(1).strip().split('\n') if skills_match else []
    return [skill.strip('▪ ').strip() for skill in skills]

def parse_professional_experience(cv_text):
    # Find the PROFESSIONAL EXPERIENCE section and extract job details
    experience_match = re.search(r'PROFESSIONAL EXPERIENCE\s+(.+?)\n\n', cv_text, re.DOTALL)
    experiences = experience_match.group(1).strip().split('\n\n') if experience_match else []
    return [exp.strip() for exp in experiences]

def parse_education(cv_text):
    # Find the EDUCATION section and extract qualifications
    education_match = re.search(r'EDUCATION\s+(.+?)\n\n', cv_text, re.DOTALL)
    education = education_match.group(1).strip().split('\n\n') if education_match else []
    return [edu.strip() for edu in education]

def create_graph_structure(personal_details, skills, experiences, education):
    # Create nodes for personal details, skills, experiences, and education
    nodes = {}
    edges = []

    # Add personal details node
    personal_node_id = 'person_0'
    nodes[personal_node_id] = personal_details

    # Add skill nodes and edges to personal node
    for i, skill in enumerate(skills):
        skill_node_id = f'skill_{i}'
        nodes[skill_node_id] = {'name': skill}
        edges.append((personal_node_id, skill_node_id))

    # Add experience nodes and edges to personal node
    for i, exp in enumerate(experiences):
        exp_node_id = f'experience_{i}'
        nodes[exp_node_id] = {'description': exp}
        edges.append((personal_node_id, exp_node_id))

    # Add education nodes and edges to personal node
    for i, edu in enumerate(education):
        edu_node_id = f'education_{i}'
        nodes[edu_node_id] = {'qualification': edu}
        edges.append((personal_node_id, edu_node_id))

    return nodes, edges

if __name__ == "__main__":
    cv_text = read_cv_text('cv_text.txt')
    personal_details = parse_personal_details(cv_text)
    skills = parse_key_skills(cv_text)
    experiences = parse_professional_experience(cv_text)
    education = parse_education(cv_text)

    # Create graph structure from parsed CV data
    nodes, edges = create_graph_structure(personal_details, skills, experiences, education)

    # Output the structured data to a file
    with open('structured_cv_data.txt', 'w') as file:
        file.write(f'Nodes:\n{nodes}\n\nEdges:\n{edges}')
