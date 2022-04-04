import React from "react";
import ReactDOM from "react-dom";

import './App.css';
import { Container, Navbar, Nav, Carousel, Row, Col, Form, Button } from "react-bootstrap";

function TitleGenerator() {

	return <div className="App">

		{/* https://react-bootstrap.github.io/components/navbar/ */}
		<Navbar>
			<Container>

				<Navbar.Brand href="#home">CloudClub</Navbar.Brand>

				<Nav className="me-auto">
					<Nav.Link href="#Latest Titles">Latest Titles</Nav.Link>
					<Nav.Link href="#Help">Help</Nav.Link>
					<Nav.Link href="#How it Works">How it Works</Nav.Link>
					<Nav.Link href="#GitHub">GitHub</Nav.Link>
				</Nav>
			</Container>
		</Navbar>


		<Container>
			<Row>
				<Col></Col>
				<Col className="justify-content-md-center">
					<center><h1>Title Generator</h1></center>
				</Col>
				<Col></Col>
			</Row>
		</Container>

		<Container>
			<Col style={{ height: 30 }}></Col>
		</Container>

		<Container>

			<Row>
				<Col></Col>
				<Col xs={6}>
					<Form>
						<Form.Group className="mb-3" controlId="exampleForm.ControlTextarea1">
							<Form.Control as="textarea" rows={15} />
						</Form.Group>
					</Form>
				</Col>
				<Col></Col>
			</Row>

		</Container>

		<Container>

			<Row>
				<Col></Col>
				<Col className="justify-content-md-center">
					<center><Button variant="primary" size="lg">Generate</Button>{' '}</center>
				</Col>
				<Col className="justify-content-md-center">
					<center><Button variant="primary" size="lg">Random</Button>{' '}</center>
				</Col>
				<Col></Col>
			</Row>

		</Container>

		<Container>
			<div className="col-xs-12" style={{ height: "100px" }}></div>
		</Container>

		<Container>
			<Row style={{ height: "100px" }}>
				<Col></Col>
				<Col className="justify-content-center"> ANSWER </Col>
				<Col></Col>
			</Row>
		</Container>

		<Container fluid>
			<Row>
				<Col className="justify-content-center">
					<img src="assets/CloudAI.png" style={{ width: "30%" }}>
					</img>
				</Col>
				<Col xs={8} className="justify-content-center">
					<img src="assets/torch.svg" width="20%">
					</img>
				</Col>
				<Col className="justify-content-center">
					<img src="assets/favicon.png" style={{ width: "30%" }}>
					</img>
				</Col>
			</Row>
		</Container>

		<div className="footer">
			<p>&copy; CloudClub 2021-2022</p>
		</div>


	</div >

}

ReactDOM.render(

	<>
		< TitleGenerator />
	</>
	,
	document.getElementById("root")
);


